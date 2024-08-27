import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class NonLocalBlock(nn.Module):
    def __init__(self, n_frames: int, r: int = 2):
        super(NonLocalBlock, self).__init__()
        self.r = r
        self.g = nn.Conv2d(3 * n_frames * r**2, 3 * n_frames * r**2, 1, 1, 0)
        self.w = nn.Conv2d(3 * n_frames * r**2, 3 * n_frames * r**2, 1, 1, 0)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        :param xs: [B, 3, T, H, W]
        :return: [B, 3, T, H, W]
        """
        h, w = xs.size(-2), xs.size(-1)
        xs_ = rearrange(xs, "b c t (h rh) (w rw) -> b (c t rh rw) h w", rh=self.r, rw=self.r, c=3)

        theta = rearrange(xs_, 'b ctr h w -> b (h w) ctr')  # B, H/R*W/R, C*T*R*R
        phi = theta.clone().permute(0, 2, 1)  # B, C*T*R*R, H/R*W/R
        g = self.g(xs_)  # B, C*T*R*R, H, W
        g = rearrange(g, 'b ctr h w -> b (h w) ctr')  # B, H/R*W/R, C*T*R*R

        f = torch.matmul(theta, phi).softmax(dim=-1)  # B, H/R*W/R, H/R*W/R
        y = torch.matmul(f, g)  # B, H/R*W/R, C*T*R*R
        y = rearrange(y, "b (h w) ctr -> b ctr h w", h=h//self.r, w=w//self.r)  # B, C*T*R*R, H/R, W/R
        y = self.w(y)  # B, C*T*R*R, H/R, W/R

        return xs + rearrange(y, "b (c t rh rw) h w -> b c t (h rh) (w rw)", rh=self.r, rw=self.r, c=3)


class ProgressiveFusionRB(nn.Module):
    def __init__(self, c_base: int, n_frames: int):
        super(ProgressiveFusionRB, self).__init__()
        self.n_frames = n_frames
        self.proj_3 = nn.Conv3d(
            c_base, c_base, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )
        self.dist_1 = nn.Conv2d(
            c_base * n_frames, c_base, kernel_size=1, stride=1, padding=0
        )
        self.agg_3 = nn.Conv3d(
            c_base * 2, c_base, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        :param xs: (B, C, T, H, W)
        :return: (B, C, T, H, W)
        """
        feat = self.proj_3(xs)
        feat = rearrange(feat, "b c t h w -> b (c t) h w")
        feat = self.dist_1(feat).unsqueeze(2).repeat(1, 1, self.n_frames, 1, 1)  # B, C, T, H, W
        feat = torch.cat([xs, feat], dim=1)  # B, C*2, T, H, W
        feat = self.agg_3(feat)  # B, C, T, H, W
        return xs + feat


class Upsample(nn.Sequential):
    def __init__(self, c_base: int, n_frames: int):
        super().__init__(
            nn.Conv2d(c_base * n_frames, 48, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(12, 12, 3, 1, 1),
            nn.PixelShuffle(2),
        )


class PFLN(nn.Module):
    def __init__(self, c_base: int, n_frames: int, n_blocks: int):
        super(PFLN, self).__init__()
        self.n_frames = n_frames

        self.non_local = NonLocalBlock(n_frames)
        self.feat_proj = nn.Conv3d(3, c_base, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2))
        self.res_blocks = nn.Sequential(
            *[ProgressiveFusionRB(c_base, n_frames) for _ in range(n_blocks)]
        )
        self.upsample = Upsample(c_base, n_frames)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        :param xs: [B, 3, T, H, W]
        :return: [B, 3, rH, rW]
        """
        t = xs.size(2)
        x_t = xs[:, :, t // 2]

        x_attn = self.non_local(xs)
        feats = self.feat_proj(x_attn)
        feats = self.res_blocks(feats)

        res = self.upsample(feats.flatten(1, 2))
        skip = F.interpolate(x_t, scale_factor=4, mode='bilinear', align_corners=False)
        return res + skip
