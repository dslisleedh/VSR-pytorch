import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class StructureDetailBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.net_s = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

        self.net_d = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, s: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s_res = self.net_s(s)
        d_res = self.net_d(d)

        return s + s_res + d_res, d + d_res + s_res


class HiddenStateAdaptation(nn.Module):
    def __init__(self, dim: int, k: int):
        super().__init__()
        self.k = k
        self.k_proj = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, k**2, 1, 1, 0),
        )

    def forward(self, lr_t: torch.Tensor, h_tm1: torch.Tensor) -> torch.Tensor:
        h, w = h_tm1.shape[-2:]
        h_tm1_unfold = F.unfold(h_tm1, self.k, padding=self.k//2)
        h_tm1_unfold = rearrange(
            h_tm1_unfold, 'b (c k2) (h w) -> b c h w k2', k2=self.k**2, h=h, w=w
        )  # (b, c, h, w, k2)

        k = self.k_proj(lr_t)
        k = rearrange(
            k, 'b k2 h w -> b 1 h w k2', k2=self.k**2, h=h, w=w
        )  # (b, 1, h, w, k2)

        sim = (h_tm1_unfold * k).sum(dim=-1).sigmoid()  # (b, c, h, w)

        return h_tm1 * sim


class StructureDetailDecomposition(nn.Module):
    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, lr_t: torch.Tensor, lr_tm1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lrs = torch.cat([lr_t, lr_tm1], dim=1)

        s = F.interpolate(
            F.interpolate(lrs, scale_factor=1/self.r, mode='bicubic', align_corners=False),
            scale_factor=self.r, mode='bicubic', align_corners=False
        )

        d = lrs - s

        return s, d


class RSDN(nn.Module):
    def __init__(
            self, dim: int = 128, scale: int = 4, n_blocks: int = 7,
            k: int = 3
    ):
        super().__init__()
        self.dim = dim
        self.scale = scale

        self.hsa = HiddenStateAdaptation(dim, k)
        self.sdd = StructureDetailDecomposition(scale)

        self.proj_s = nn.Sequential(
            nn.Conv2d(3 * 2 + 3 * scale**2 + dim, dim, 3, padding=1),  # S_t, S_tm1, S_tm1_hat, h_tm1
            nn.ReLU(),
        )
        self.proj_d = nn.Sequential(
            nn.Conv2d(3 * 2 + 3 * scale**2 + dim, dim, 3, padding=1),  # D_t, D_tm1, D_tm1_hat, h_tm1
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            StructureDetailBlock(dim) for _ in range(n_blocks)
        ])

        self.to_rgb_s = nn.Conv2d(dim, 3 * self.scale**2, 3, padding=1)
        self.to_rgb_d = nn.Conv2d(dim, 3 * self.scale**2, 3, padding=1)
        self.to_rgb_cat = nn.Conv2d(dim * 2, 3 * self.scale**2, 3, padding=1)
        self.to_h_sd = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.ReLU(),
        )

        self.upsample = nn.PixelShuffle(scale)

    def forward(self, lr_t: torch.Tensor, references: dict | None = None) -> tuple[torch.Tensor, dict]:
        b, c, h, w = lr_t.shape
        lr_tm1 = references.get('lr_tm1') if references else lr_t.clone()
        h_tm1 = references.get('h_tm1') if references else torch.zeros(b, self.dim, h, w).to(lr_t.device)
        s_tm1_hat = references.get('s_tm1_hat') if references else torch.zeros(b, 3 * self.scale**2, h, w).to(lr_t.device)
        d_tm1_hat = references.get('d_tm1_hat') if references else torch.zeros(b, 3 * self.scale**2, h, w).to(lr_t.device)

        s_t, d_t = self.sdd(lr_t, lr_tm1)
        h_t = self.hsa(lr_t, h_tm1)

        s = self.proj_s(torch.cat([s_t, s_tm1_hat, h_t], dim=1))  # s_t contains s_t and s_tm1
        d = self.proj_d(torch.cat([d_t, d_tm1_hat, h_t], dim=1))  # d_t contains d_t and d_tm1

        for block in self.blocks:
            s, d = block(s, d)

        h_sd = self.to_h_sd(s + d)

        s_hat = self.to_rgb_s(s)
        d_hat = self.to_rgb_d(d)
        rgb_hat = self.to_rgb_cat(torch.cat([s, d], dim=1))

        s_hat_up = self.upsample(s_hat)
        d_hat_up = self.upsample(d_hat)
        rgb_hat_up = self.upsample(rgb_hat)

        res_reference_dict = {
            'h_sd': h_sd,
            's_hat': s_hat,
            'd_hat': d_hat,
            's_hat_up': s_hat_up,
            'd_hat_up': d_hat_up,
        }

        return rgb_hat_up, res_reference_dict
