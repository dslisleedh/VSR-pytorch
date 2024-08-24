import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from typing import Sequence


class MotionCompensator(nn.Module):
    def __init__(self):
        super().__init__()
        self.coarse_flow = nn.Sequential(*[
            nn.Conv2d(2, 24, 5, 2, 2),  # 2 frames with y-channel
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, 1, 1),
            nn.Tanh(),
            nn.PixelShuffle(4)
        ])

        self.fine_flow = nn.Sequential(*[
            nn.Conv2d(5, 24, 5, 2, 2),  # 2 frames with y-channel + 2 flows + 1 warped frame(t+-1)
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 8, 3, 1, 1),
            nn.Tanh(),
            nn.PixelShuffle(2)
        ])

    def forward(self, x_t: torch.Tensor, x_t1: torch.Tensor) -> Sequence[torch.Tensor]:
        """
        :param xs: [B, 2, H, W], 2 frames
        :return:
            x_mc: [B, 1, H, W], motion compensated frame
            flow: [B, 2, H, W], flow
        """
        h, w = x_t1.shape[-2:]
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w)
        ), dim=-1).unsqueeze(0).to(x_t1.device).flip(-1)

        xs = torch.cat([x_t, x_t1], dim=1)

        coarse_flow = self.coarse_flow(xs)
        coarse_flow[:, 0] = coarse_flow[:, 0] / w
        coarse_flow[:, 1] = coarse_flow[:, 1] / h

        x_mc_c = F.grid_sample(
            x_t1, torch.clamp(grid - coarse_flow.permute(0, 2, 3, 1), -1, 1),
            align_corners=True, mode='bilinear'
        )

        fine_flow = self.fine_flow(torch.cat([xs, coarse_flow, x_mc_c], dim=1))
        fine_flow[:, 0] = fine_flow[:, 0] / w
        fine_flow[:, 1] = fine_flow[:, 1] / h

        flow = coarse_flow + fine_flow
        x_mc = F.grid_sample(
            x_t1, torch.clamp(grid - flow.permute(0, 2, 3, 1), -1, 1),
            align_corners=True, mode='bilinear'
        )
        return x_mc, flow


class ESPCN(nn.Sequential):
    def __init__(
        self,  n_feat: int = 24, n_layers: int = 6,
        scale: int = 4
    ):
        self.scale = scale

        layers = []

        # early fusion
        layers.append(nn.Conv2d(3, n_feat, 3, 1, 1))
        layers.append(nn.ReLU())

        # hidden layers
        for _ in range(n_layers):
            layers.append(nn.Conv2d(n_feat, n_feat, 3, 1, 1))
            layers.append(nn.ReLU())

        # upsample
        layers.append(nn.Conv2d(n_feat, n_feat * scale ** 2, 3, 1, 1))
        layers.append(nn.PixelShuffle(scale))

        super().__init__(*layers)


class VESPCN(nn.Module):
    def __init__(
        self, n_feat: int = 24, n_layers: int = 6,
        scale: int = 4
    ):
        super().__init__()
        self.motion_compensator = MotionCompensator()
        self.espcn = ESPCN(n_feat, n_layers, scale)

        self.initialize()

    def forward(self, xs: torch.Tensor):
        x_tm1, x_t, x_tp1 = xs.chunk(3, dim=1)
        x_tm1_mc, flow_tm1 = self.motion_compensator(x_t, x_tm1)
        x_tp1_mc, flow_tp1 = self.motion_compensator(x_t, x_tp1)

        xs = torch.cat([x_tm1_mc, x_t, x_tp1_mc], dim=1)
        x_sr = self.espcn(xs)

        if self.training:
            return x_sr, flow_tm1, flow_tp1
        else:
            return x_sr

    def initialize(self):
        for w in self.espcn.parameters():
            if w.dim() > 1:
                nn.init.orthogonal_(w, gain=2**0.5)
            else:
                nn.init.zeros_(w)

        for w in self.motion_compensator.parameters():
            if w.dim() > 1:
                nn.init.orthogonal_(w, gain=2**0.5)
            else:
                nn.init.zeros_(w)
