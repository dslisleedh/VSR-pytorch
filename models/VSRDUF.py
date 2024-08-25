import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def dynamic_upsampling(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """
    :param x: LR_t [B, 3, H, W]
    :param f: Filter [H, W, 25, r^2]
    :return: Upsampled tensor [B, 3, H*r, W*r]
    """
    h, w = x.shape[2], x.shape[3]
    scale = int(f.shape[-1] ** 0.5)
    # b 3 h w -> (b 3) 1 h w
    x = rearrange(x, 'b c h w -> (b c) 1 h w')
    # unfold
    x = F.pad(x, (2, 2, 2, 2), mode='replicate')
    x = F.unfold(x, kernel_size=5)
    x = rearrange(x, 'b kk (h w) -> b h w kk', h=h, w=w)
    # matmul
    f = f.softmax(dim=-2)
    x = torch.einsum('bhwc, hwcr -> bhwr', x, f)
    # rearrange
    x = rearrange(x, '(b rgb) h w (rh rw) -> b rgb (h rh) (w rw)', rh=scale, rw=scale, rgb=3)
    return x
