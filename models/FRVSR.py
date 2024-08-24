import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Sequence


class FNet(nn.Sequential):
    def __init__(self):
        layers = []
        # H W
        layers.extend([
            nn.Conv2d(6, 32, 3, 1 ,1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        ])
        # H/2 W/2
        layers.extend([
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        ])
        # H/4 W/4
        layers.extend([
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2),
        ])
        # H/8 W/8
        layers.extend([
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        ])
        # H/4 W/4
        layers.extend([
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        ])
        # H/2 W/2
        layers.extend([
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        ])
        # H W
        layers.extend([
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 2, 3, 1, 1),
            nn.Tanh(),
        ])
        super(FNet, self).__init__(*layers)


class ResidualBlock(nn.Sequential):
    def __init__(self):
        super().__init__(*[
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)


class Upsampler(nn.Sequential):
    def __init__(self, scale: int):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(
                    nn.ConvTranspose2d(64, 64, 4, 2, 1),
                )
                m.append(
                    nn.ReLU(inplace=True),
                )
        else:
            raise NotImplementedError

        m.append(
            nn.Conv2d(64, 3, 3, 1, 1),
        )
        super().__init__(*m)


class SRNet(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3 + 3 * scale**2, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock() for _ in range(10)])
        self.tail = Upsampler(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


class FRVSR(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.fnet = FNet()
        self.srnet = SRNet(scale)

        self.scale = scale

    def forward(self, x_t, x_tm1=None, x_est_tm1=None) -> torch.Tensor | Sequence[torch.Tensor]:
        x_tm1 = x_tm1 if x_tm1 else torch.zeros_like(x_t)  # black image if t=0
        x_est_tm1 = x_est_tm1 if x_est_tm1 else torch.zeros_like(x_t)

        h, w = x_t.shape[-2:]
        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w)
        ), dim=-1).unsqueeze(0).to(x_t.device).flip(-1).permute(0, 3, 1, 2)
        flow_lr = self.fnet(torch.cat([x_t, x_tm1], dim=1))
        flow_lr[:, 0] = flow_lr[:, 0] / w
        flow_lr[:, 1] = flow_lr[:, 1] / h
        flow_lr = torch.clamp(grid - flow_lr, -1, 1)
        flow_hr = F.interpolate(
            flow_lr, scale_factor=self.scale, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

        x_est = F.grid_sample(x_est_tm1, flow_hr, mode='bilinear', padding_mode='border', align_corners=False)
        x_est = F.pixel_unshuffle(x_est, self.scale)

        inp = torch.cat([x_t, x_est], dim=1)
        x_sr = self.srnet(inp)

        if self.training:
            return x_sr, flow_lr.permute(0, 2, 3, 1)
        else:
            return x_sr
