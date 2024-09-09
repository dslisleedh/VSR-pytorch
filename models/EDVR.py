import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from functools import partial
from typing import Sequence

from layers.deformable_conv_v2 import DeformableConv2dPack


upsample_only_spatial = partial(F.interpolate, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat: int = 64, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


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


class PCDAlignment(nn.Module):
    def __init__(self, c_base: int, n_groups: int):
        super().__init__()

        self.offset_l3 = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_base, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dcn_l3 = DeformableConv2dPack(c_base, n_groups)

        self.offset_l2_proj = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.offset_l2_agg = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_base, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dcn_l2 = DeformableConv2dPack(c_base, n_groups)
        self.feat_l2 = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.offset_l1_proj = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.offset_l1_agg = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_base, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dcn_l1 = DeformableConv2dPack(c_base, n_groups)
        self.feat_l1 = nn.Conv2d(c_base * 2, c_base, 3, 1, 1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.cas_offset = nn.Sequential(
            nn.Conv2d(c_base * 2, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_base, c_base, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.cas_dcn = DeformableConv2dPack(c_base, n_groups)
        self.cas_dcn_act = nn.LeakyReLU(0.1, inplace=True)

    def forward(
            self, x_t_l1: torch.Tensor, x_t_l2: torch.Tensor, x_t_l3: torch.Tensor,
            x_nearby_l1: torch.Tensor, x_nearby_l2: torch.Tensor, x_nearby_l3: torch.Tensor
    ) -> torch.Tensor:
        # L3
        offset = self.offset_l3(
            torch.cat([x_nearby_l3, x_t_l3], dim=1)
        )
        feat = self.dcn_l3(x_nearby_l3, offset)
        upsampled_offset = self.up(offset) * 2
        upsampled_feat = self.up(feat)

        # L2
        offset = self.offset_l2_proj(
            torch.cat([x_nearby_l2, x_t_l2], dim=1)
        )
        offset = self.offset_l2_agg(
            torch.cat([offset, upsampled_offset], dim=1)
        )
        feat = self.dcn_l2(x_nearby_l2, offset)
        feat = self.feat_l2(torch.cat([feat, upsampled_feat], dim=1))
        upsampled_offset = self.up(offset) * 2
        upsampled_feat = self.up(feat)

        # L1
        offset = self.offset_l1_proj(
            torch.cat([x_nearby_l1, x_t_l1], dim=1)
        )
        offset = self.offset_l1_agg(
            torch.cat([offset, upsampled_offset], dim=1)
        )
        feat = self.dcn_l1(x_nearby_l1, offset)
        feat = self.feat_l1(torch.cat([feat, upsampled_feat], dim=1))

        # Cascading
        offset = self.cas_offset(torch.cat([feat, x_t_l1], dim=1))
        feat = self.cas_dcn_act(self.cas_dcn(feat, offset))
        return feat


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


class EDVR(nn.Module):
    def __init__(
            self, c_base: int = 64, n_frames: int = 5, n_deform_groups: int = 8,
            n_extra_blocks: int = 5, n_recon_blocks: int = 10
    ):
        super().__init__()
        self.pre_deblur = nn.Sequential(
            PreDeblur(c_base),
            nn.Conv3d(c_base, c_base, 1, 1, 0)
        )
        self.to_l1 = nn.Sequential(*[
            ResidualBlockNoBN3D(c_base) for _ in range(n_extra_blocks)
        ])
        self.to_l2 = nn.Sequential(
            nn.Conv3d(c_base, c_base, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c_base, c_base, (1, 3, 3), 1, (0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.to_l3 = nn.Sequential(
            nn.Conv3d(c_base, c_base, (1, 3, 3), (1, 2, 2), (0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(c_base, c_base, (1, 3, 3), 1, (0, 1, 1)),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.pcd_alignment = PCDAlignment(c_base, n_deform_groups)

        self.tsa_fusion = TSAFusion(c_base, n_frames)

        self.recon = nn.Sequential(*[
            ResidualBlockNoBN(c_base) for _ in range(n_recon_blocks)
        ])
        self.upsample = nn.Sequential(
            nn.Conv2d(c_base, c_base * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(c_base, 64 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        :param xs: [B, 3, 5, T, H, W]
        :return: [B, 3, 4H, 4W]
        """
        x_t = xs[:, :, 2]
        t = xs.size(2)

        # Generate pyramid features
        feat_l1 = self.pre_deblur(xs)
        feat_l1 = self.to_l1(feat_l1)
        feat_l2 = self.to_l2(feat_l1)
        feat_l3 = self.to_l3(feat_l2)

        # PCD alignment
        feat = []
        for i in range(t):
            feat.append(
                self.pcd_alignment(
                    feat_l1[:, :, 2], feat_l2[:, :, 2], feat_l3[:, :, 2],
                    feat_l1[:, :, i], feat_l2[:, :, i], feat_l3[:, :, i]
                )
            )
        feat = torch.stack(feat, dim=2)  # B C T H W

        # TSA fusion
        feat = self.tsa_fusion(feat)

        # Reconstruction and upsample
        feat = self.recon(feat)
        out = self.upsample(feat) + F.interpolate(x_t, scale_factor=4, mode='bilinear', align_corners=False)
        return out
