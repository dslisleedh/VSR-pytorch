import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


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
