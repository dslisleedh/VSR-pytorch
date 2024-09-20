import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.SPyNet import BidirectionalSPyNet
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
                flow = flow_backward[:, :, i]
                h = flow_warp(h, flow)
            h = self.backward_path(torch.cat([h, lr], dim=1))
            backwards.append(h)
        backwards = backwards[::-1]

        forwards = []
        h = torch.zeros(B, self.n_feats, H, W).to(lrs.device)
        for i in range(T):
            lr = lrs[:, :, i]
            if i != 0:
                flow = flow_forward[:, :, i-1]
                h = flow_warp(h, flow)
            h = self.forward_path(torch.cat([h, lr], dim=1))
            forwards.append(h)

        srs = []
        for b, f in zip(forwards, backwards):
            srs.append(self.recon(torch.cat([b, f], dim=1)))

        return torch.stack(srs, dim=2)
