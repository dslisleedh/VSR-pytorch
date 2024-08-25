import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class DenseBlock(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, reduce_temporal: bool):
        # No intermediate expansion
        super().__init__(*[
            nn.BatchNorm3d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_in, c_in, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_in, c_out, kernel_size=3, stride=1, padding=(0, 1, 1) if reduce_temporal else 1),
        ])


class GenerationNetwork(nn.Module):
    def __init__(self, c_base: int, c_growth: int, n_blocks: int):
        super().__init__()
        # frames always 7
        # reduce temporal dimension at the last 3 blocks
        self.to_feat = nn.Conv3d(
            3, c_base, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )
        self.blocks_refine = nn.ModuleList([
            DenseBlock(c_base + i * c_growth, c_growth, False)
            for i in range(n_blocks - 3)
        ])
        self.blocks_reduce = nn.ModuleList([
            DenseBlock(c_base + (n_blocks - 3 + i) * c_growth, c_growth, True)
            for i in range(3)
        ])
        c_out = c_base + n_blocks * c_growth
        self.to_out = nn.Sequential(
            nn.BatchNorm3d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(c_out, c_out, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_feat(x)
        for block in self.blocks_refine:
            x = torch.cat([x, block(x)], dim=1)
        for block in self.blocks_reduce:
            x = torch.cat([x[:, :, 1:-1, :, :], block(x)], dim=1)
        x = self.to_out(x)[:, :, 0, :, :]
        return x


# Residual prediction network
class RNet(nn.Sequential):
    def __init__(self, c_in: int, scale: int):
        super().__init__(*[
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_in, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, 3 * scale ** 2, 1, 1, 0),
            nn.PixelShuffle(scale)
        ])


# Dynamic filter prediction network
class FNet(nn.Sequential):
    def __init__(self, c_in, scale):
        super().__init__(*[
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_in * 2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in * 2, 25 * scale ** 2, 1, 1, 0)
        ])  # No softmax here -> dynamic_upsampling function


def dynamic_upsampling(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    :param x: LR_t [B, 3, H, W]
    :param f: Filter [B, H, W, 25, r^2]
    :return: Upsampled tensor [B, 3, H*r, W*r]
    """
    h, w = x.shape[2], x.shape[3]
    scale = int(f.shape[-1] ** 0.5)
    # unfold
    x = F.pad(x, (2, 2, 2, 2), mode='replicate')
    x = F.unfold(x, kernel_size=5)
    x = rearrange(x, 'b (rgb kk) (h w) -> rgb b h w kk', rgb=3, h=h, w=w)
    # matmul
    f = f.softmax(dim=-2)
    x = torch.einsum('cbhwk, bhwkr -> cbhwr', x, f)
    # rearrange
    x = rearrange(x, 'rgb b h w (rh rw) -> b rgb (h rh) (w rw)', rh=scale, rw=scale, rgb=3)
    return x


class VSRDUF(nn.Module):
    def __init__(self, c_base: int, c_growth: int, n_blocks: int, scale: int):
        super().__init__()
        self.scale = scale
        self.gnet = GenerationNetwork(c_base, c_growth, n_blocks)
        self.rnet = RNet(c_base + n_blocks * c_growth, scale)
        self.fnet = FNet(c_base + n_blocks * c_growth, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.gnet(x)
        r = self.rnet(feat)
        f = self.fnet(feat)
        f = rearrange(f, 'b (k2 r2) h w -> b h w k2 r2', k2=25, r2=self.scale ** 2)
        return dynamic_upsampling(x[:, :, 3, :, :], f) + r  # t frame
    

