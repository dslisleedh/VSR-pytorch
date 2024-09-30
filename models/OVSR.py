import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

# from models.PFLN import ProgressiveFusionRB
from layers.upsample import AdHocSPConv

from typing import Literal


class ProgressiveFusionRB(nn.Module):  # For B(TC)HW type input
    def __init__(self, c_base: int, n_frames: int = 3):
        super(ProgressiveFusionRB, self).__init__()
        self.n_frames = n_frames
        self.proj_3 = nn.Conv2d(
            c_base * n_frames, c_base * n_frames, kernel_size=3, stride=1, padding=1, groups=n_frames
        )
        self.agg_3 = nn.Conv2d(
            c_base * n_frames * 2, c_base * n_frames, kernel_size=3, stride=1, padding=1, groups=n_frames
        )
        self.dist_1 = nn.Conv2d(
            c_base * n_frames, c_base, kernel_size=1, stride=1, padding=0
        )

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """
        :param xs: [B (3 C) H W]
        :return: [B (3 C) H W]
        """
        feat = self.proj_3(xs)  # B, T*C, H, W
        feat_dist = self.dist_1(feat).unsqueeze(1).repeat(1, self.n_frames, 1, 1, 1)  # B, T, C, H, W
        feat = rearrange(feat, 'b (t c) h w -> b t c h w', t=self.n_frames)
        feat = torch.cat([feat, feat_dist], dim=2).flatten(1, 2)  # B, T*C*2, H, W
        feat = self.agg_3(feat)
        return xs + feat


class NetP(nn.Module):
    def __init__(
            self, c_base: int, n_blocks: int,
            upscaling_factor: int, n_frames: int = 3
    ):
        super().__init__()
        self.n_frames = n_frames
        self.proj_ref = nn.Conv2d(3, c_base, 3, 1, 1)  # Image at time t
        self.proj_sup = nn.Conv2d(3 * 2, c_base, 3, 1, 1)  # Image at time t-1 and t+1
        self.blocks = nn.Sequential(*[
            ProgressiveFusionRB(c_base, n_frames)
            for _ in range(n_blocks)
        ])
        self.agg = nn.Sequential(
            nn.Conv2d(c_base * n_frames, c_base, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsampler = AdHocSPConv(c_base, upscaling_factor)

    def forward(self, lrs: torch.Tensor, h_tm1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param lrs: [B, 3, 3, H, W]
        :param h_tm1: [B, C, H, W]
        :return: [B, 3, H * upscaling_factor, W * upscaling_factor], [B, C, H, W]
        """
        ref = self.proj_ref(lrs[:, :, 1])  # B C H W
        sup = self.proj_sup(
            torch.cat([lrs[:, :, 0], lrs[:, :, 2]], dim=1)
        )
        inp = torch.cat([ref, sup, h_tm1], dim=1)
        h = self.agg(self.blocks(inp))
        sr = self.upsampler(h)
        return sr, h


class NetS(nn.Module):
    def __init__(
            self, c_base: int, n_blocks: int,
            upscaling_factor: int, n_frames: int = 3
    ):
        super().__init__()
        self.n_frames = n_frames
        self.proj = nn.Sequential(
            nn.Conv2d((c_base + 3) * n_frames, c_base * n_frames, 3, 1, 1, groups=n_frames),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.blocks = nn.Sequential(*[
            ProgressiveFusionRB(c_base, n_frames)
            for _ in range(n_blocks)
        ])
        self.agg = nn.Sequential(
            nn.Conv2d(c_base * n_frames, c_base, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.upsampler = AdHocSPConv(c_base, upscaling_factor)

    def forward(self, lrs: torch.Tensor, hs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param lrs: [B, 3, 3, H, W]
        :param hs: [B, C, 3, H, W]
        :return: [B, 3, H * upscaling_factor, W * upscaling_factor], [B, C, H, W]
        """
        inp = torch.cat([lrs, hs], dim=1)
        inp = self.proj(inp.transpose(1, 2).flatten(1, 2).contiguous())  # B (T C) H W
        h = self.agg(self.blocks(inp))  # B C H W
        sr = self.upsampler(h)  # B 3 rH rW
        return sr, h


class OVSR(nn.Module):
    def __init__(
            self, c_base: int, n_p_blocks: int, n_s_blocks: int,
            upscaling_factor: int, net_type: Literal['LOVSR', 'GOVSR']
    ):
        super().__init__()
        # scaling factor(alpha) is used in loss function
        self.net_type = net_type

        self.net_p = NetP(c_base, n_p_blocks, upscaling_factor)
        self.net_s = NetS(c_base, n_s_blocks, upscaling_factor)

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        """
        :param lrs: [B, 3, T, H, W]
        :return: [B, 3, T, H * upscaling_factor, W * upscaling_factor]
        """
        B, _, T, H, W = lrs.shape

        # precursor step
        h_tm1 = torch.zeros(B, 64, H, W, device=lrs.device)
        h_p = []
        sr_p = []
        lrs = F.pad(lrs, (0, 0, 0, 0, 1, 1), 'replicate')
        if self.net_type == 'GOVSR':
            lrs = torch.flip(lrs, [2])
        for t in range(1, T + 1):
            cur_lr = lrs[:, :, t - 1:t + 2]  # B 3 3 H W
            sr_t, h_tm1 = self.net_p(cur_lr, h_tm1)
            sr_p.append(sr_t)
            h_p.append(h_tm1)
        sr_p = torch.stack(sr_p, dim=2)
        if self.net_type == 'GOVSR':
            lrs = torch.flip(lrs, [2])
            sr_p = torch.flip(sr_p, [2])
            h_p = h_p[::-1]

        # successor step
        h_tm1 = torch.zeros(B, 64, H, W, device=lrs.device)
        sr_s = []
        for t in range(1, T + 1):
            cur_lr = lrs[:, :, t - 1:t + 2]
            h_p_t = h_p[t - 1]
            h_p_tp1 = h_p[t] if t < T else h_p[t - 1]
            hs = torch.stack([h_tm1, h_p_t, h_p_tp1], dim=2)
            sr_t, h_tm1 = self.net_s(cur_lr, hs)

            sr_s.append(sr_t)

        sr_s = torch.stack(sr_s, dim=2)
        return sr_s + sr_p
