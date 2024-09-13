import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.SPyNet import SPyNet
from utils.warp import flow_warp


class ResBlock(nn.Sequential):
    def __init__(self, n_feats: int):
        super().__init__(
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)


class ResBlocksInputConv(nn.Sequential):
    def __init__(self, n_feat_in: int, n_feats: int, n_blocks: int):
        super().__init__(
            nn.Conv2d(n_feat_in, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            *[ResBlock(n_feats) for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)


class BidirectionalSPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.snet = SPyNet()

    def forward(self, lrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param lrs: [B, C, T, H, W]
        :return: [B, 2, T-1, H, W], [B, 2, T-1, H, W]
        """
        t = lrs.size(2)
        x_1 = rearrange(
            lrs[:, :, :-1], 'b c t h w -> (b t) c h w'
        )
        x_2 = rearrange(
            lrs[:, :, 1:], 'b c t h w -> (b t) c h w'
        )

        flow_forward = rearrange(
            self.snet(x_1, x_2), '(b t) uv h w -> b uv t h w', t=t-1
        )
        flow_backward = rearrange(
            self.snet(x_2, x_1), '(b t) uv h w -> b uv t h w', t=t-1
        )

        return flow_forward, flow_backward


class BasicVSR(nn.Module):
    def __init__(
            self, n_feats: int, n_blocks: int
    ):
        super().__init__()
        self.n_feats = n_feats

        self.snet = BidirectionalSPyNet()

        self.backward_path = ResBlocksInputConv(n_feats + 3, n_feats, n_blocks)
        self.forward_path = ResBlocksInputConv(n_feats + 3, n_feats, n_blocks)

        self.recon = nn.Sequential(
            nn.Conv2d(n_feats * 2, n_feats, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, 64 * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        """
        :param lrs: [B, C, T, H, W]
        :return: [B, C, T, 4H, 4W]
        """
        flow_forward, flow_backward = self.snet(lrs)

        B, _, T, H, W = lrs.size()

        backwards = []
        h = torch.zeros(B, self.n_feats, H, W).to(lrs.device)
        for i in range(T-1, -1, -1):
            lr = lrs[:, :, i]
            if i != T-1:
                flow = flow_forward[:, :, i]
                h = flow_warp(h, flow)
            h = self.backward_path(torch.cat([h, lr], dim=1))
            backwards.append(h)
        backwards = backwards[::-1]

        forwards = []
        h = torch.zeros(B, self.n_feats, H, W).to(lrs.device)
        for i in range(T):
            lr = lrs[:, :, i]
            if i != 0:
                flow = flow_backward[:, :, i-1]
                h = flow_warp(h, flow)
            h = self.backward_path(torch.cat([h, lr], dim=1))
            forwards.append(h)

        srs = []
        for b, f in zip(forwards, backwards):
            srs.append(self.recon(torch.cat([b, f], dim=1)))

        return torch.stack(srs, dim=2)
