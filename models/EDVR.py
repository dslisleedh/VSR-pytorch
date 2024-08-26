import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import deform_conv2d

from einops import rearrange

from functools import partial


upsample_only_spatial = partial(F.interpolate, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)


class ResidualBlockNoBN3D(nn.Module):
    def __init__(self, num_feat: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class PreDeblur(nn.Module):
    def __init__(self, c_base: int):
        super().__init__()
        self.proj_l1 = nn.Conv3d(3, c_base, (1, 3, 3), 1, (0, 1, 1))

        self.proj_l2 = nn.Sequential(
            nn.Conv3d(c_base, c_base, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.proj_l3 = nn.Sequential(
            nn.Conv3d(c_base, c_base, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.refine_l3 = nn.Conv3d(c_base, c_base, (1, 3, 3), 1, (0, 1, 1))
        self.refine_l2 = nn.Conv3d(c_base, c_base, (1, 3, 3), 1, (0, 1, 1))
        self.refine_l2pl3 = nn.Conv3d(c_base, c_base, (1, 3, 3), 1, (0, 1, 1))
        self.refine_l1_bef_add = nn.ModuleList([
            ResidualBlockNoBN3D(c_base) for _ in range(2)
        ])
        self.refine_l1_aft_add = nn.ModuleList([
            ResidualBlockNoBN3D(c_base) for _ in range(3)
        ])

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        :param xs: [B, 3, T, H, W]
        :return: [B, C, T, H, W]
        """
        feat_l1 = self.proj_l1(xs)
        feat_l2 = self.proj_l2(feat_l1)
        feat_l3 = self.refine_l3(self.proj_l3(feat_l2))
        feat_l2 = self.refine_l2pl3(
            self.refine_l2(feat_l2) + upsample_only_spatial(feat_l3)
        )
        for m in self.refine_l1_bef_add:
            feat_l1 = m(feat_l1)
        feat_l1 = feat_l1 + upsample_only_spatial(feat_l2)
        for m in self.refine_l1_aft_add:
            feat_l1 = m(feat_l1)
        return feat_l1


class DeformableConv2dPack(nn.Module):
    def __init__(self, c_base: int, deformable_groups: int):
        super().__init__()
        self.register_parameter(
            "weight", nn.Parameter(torch.randn(c_base, c_base, 3, 3) * 0.02)
        )
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(c_base))
        )
        self.offset_mask_conv = nn.Conv2d(
            c_base, 3 * deformable_groups * 3 * 3, 3, 1, 1  # offset + mask * groups * kh * kw
        )
        self.init_weights()

    def init_weights(self):
        self.offset_mask_conv.weight.data.zero_()
        self.offset_mask_conv.bias.data.zero_()

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (B, C, H, W)
        :param feat: Tensor of shape (B, C, H, W)
        :return: Tensor of shape (B, C, H, W)
        """
        out = self.offset_mask_conv(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(x, offset, self.weight, padding=(1, 1), mask=mask)

    def extra_repr(self):
        return f"(weight): {self.weight.shape} \n(bias): {self.bias.shape}"


class PCDAlignment(nn.Module):
    def __init__(self, c_base: int, n_groups: int):
        super().__init__()


class TSAFusion(nn.Module):
    def __init__(self, c_base: int, n_frames: int):
        super().__init__()
        self.c_base = c_base
        self.n_frames = n_frames
        self.center_idx = n_frames // 2

        # Temporal attention
        self.temporal_proj_t = nn.Conv2d(c_base, c_base, 3, 1, 1)
        self.temporal_proj = nn.Conv3d(c_base, c_base, (1, 3, 3), 1, (0, 1, 1))

        # Fuse Temporal attention
        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(c_base * n_frames, c_base, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Spatial attention
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mp = nn.MaxPool2d(3, stride=2, padding=1)
        self.ap = nn.AvgPool2d(3, stride=2, padding=1)

        self.spatial_proj_l2 = nn.Sequential(
            nn.Conv2d(c_base, c_base, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.spatial_agg_l2 = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.spatial_proj_l3 = nn.Sequential(
            nn.Conv2d(c_base, c_base, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.spatial_agg_l3 = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.spatial_refine_l3 = nn.Sequential(
            nn.Conv2d(c_base, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.spatial_refine_l2 = nn.Sequential(
            nn.Conv2d(c_base, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.spatial_refine_l2pl3 = nn.Sequential(
            nn.Conv2d(c_base, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.spatial_attn_mul = nn.Conv2d(c_base, c_base, 3, 1, 1)
        self.spatial_attn_add = nn.Sequential(
            nn.Conv2d(c_base, c_base, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_base, c_base, 1, 1, 0),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        :param xs: Tensor of shape (B, C, T, H, W)
        :return: Tensor of shape (B, C, H, W)
        """
        # Temporal attention
        x_t_embedding = self.temporal_proj_t(xs[:, :, self.center_idx]).unsqueeze(2)
        xs_embedding = self.temporal_proj(xs)

        attn = torch.sigmoid(
            torch.sum(
                x_t_embedding * xs_embedding, dim=1, keepdim=True
            )
        )  # B, 1, T, H, W
        xs = (xs * attn).flatten(1, 2)  # B C T H W -> B C T*H*W

        feat = self.temporal_fusion(xs)  # B C H W

        # Spatial attention
        attn_l2 = self.spatial_proj_l2(feat)  # B C H W
        attn_l2 = self.spatial_agg_l2(
            torch.cat([self.mp(attn_l2), self.ap(attn_l2)], dim=1)
        )  # B C H/2 W/2
        attn_l3 = self.spatial_proj_l3(attn_l2)  # B C H/2 W/2
        attn_l3 = self.spatial_agg_l3(
            torch.cat([self.mp(attn_l3), self.ap(attn_l3)], dim=1)
        )  # B C H/4 W/4
        attn_l3 = self.spatial_refine_l3(attn_l3)  # B C H/4 W/4
        attn_l2 = self.spatial_refine_l2pl3(self.spatial_refine_l2(attn_l2) + self.up(attn_l3))  # B C H/2 W/2
        attn_l1 = self.spatial_attn_mul(self.up(attn_l2))  # B C H W
        attn_mul = torch.sigmoid(attn_l1)
        attn_add = self.spatial_attn_add(attn_l1)

        feat = feat * attn_mul * 2 + attn_add  # *2 for initialize attn as 1.
        return feat


class Reconstruction(nn.Module):
    ...


class EDVR(nn.Module):
    ...
