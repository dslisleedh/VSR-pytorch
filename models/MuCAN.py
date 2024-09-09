import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


"""
Q1. how to fuse the attention features from previous level?
Q2. Why output size is denoted as H W S2 in AggregationUnit?
Q3. How to compute AggregationUnit on M^0 and M^s? There's spatial resolution difference since avg pool with s=2.

Solutions(not official but my conjecture):
A1. Upsample and add it to both features.
A2. mayby c is ommited in the formula. so, output is H W C.
A3. Upsample M^s to the same size as M^0 and then compute AggregationUnit.
"""


class AggregationUnit(nn.Module):
    def __init__(
            self, n_feat: int, d: int, patch_size: int = 3, k: int = 4
    ):
        super().__init__()
        self.d = d  # 3, 5, or 7
        self.patch_size = patch_size
        self.k = k

        self.aggregator = nn.Conv2d(k, 1, 1)
        self.w_proj = nn.Conv2d(n_feat * 2, n_feat * patch_size ** 2, 1)

    def forward(self, feat_t: torch.Tensor, feat_tm1: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat_t.shape
        feat_t_unfold = F.unfold(feat_t, self.patch_size, padding=self.patch_size // 2)
        feat_t_unfold = rearrange(
            feat_t_unfold, 'b cp2 (h w) -> b cp2 h w', h=h, w=w
        )

        feat_tm1_unfold = F.unfold(feat_tm1, self.patch_size, padding=self.patch_size // 2)
        feat_tm1_unfold = rearrange(
            feat_tm1_unfold, 'b cp2 (h w) -> b cp2 h w', h=h, w=w
        )

        # compute similarity between patches inner displacement(self.d) to reduce computation
        feat_tm1_displaced = F.unfold(
            feat_tm1_unfold, self.d, stride=1, padding=self.d // 2
        )
        feat_tm1_displaced = rearrange(
            feat_tm1_displaced, 'b (cp2 d) (h w) -> b cp2 h w d', d=self.d ** 2, h=h, w=w
        )

        feat_t_unfold = feat_t_unfold.unsqueeze(-1)

        sim = F.normalize(feat_t_unfold, dim=1) * F.normalize(feat_tm1_displaced, dim=1)  # b cp2 h w d
        sim = torch.sum(sim, dim=1, keepdim=True)
        _, idx = torch.topk(sim, self.k, dim=-1)
        feat_t_related = torch.gather(feat_tm1_displaced, -1, idx.repeat(1, c * self.patch_size ** 2, 1, 1, 1))

        feat_t_related = rearrange(
            feat_t_related, 'b (c ph pw) h w k -> (b h w c) k ph pw',
            b=b, c=c, ph=self.patch_size, pw=self.patch_size
        )
        feat_t_related = rearrange(
            self.aggregator(feat_t_related), '(b h w c) 1 ph pw -> b c h w (ph pw)',
            b=b, h=h, w=w, c=c
        )

        w = self.w_proj(torch.cat([feat_t, feat_tm1], dim=1))  # b (c*p2) h w
        w = rearrange(
            w, 'b (c p2) h w -> b c h w p2', c=c
        )

        feat_t_related = torch.sum(feat_t_related * w, dim=-1)  # b c h w
        return feat_t_related


class TMCAM(nn.Module):
    def __init__(
            self, n_feat: int, n_frames: int,
            ds: list = [3, 5, 7], patch_size: int = 3, k: int = 4
    ):
        super().__init__()
        self.n_frames = n_frames

        self.au_l3 = AggregationUnit(n_feat, ds[0], patch_size, k)
        self.au_l2 = AggregationUnit(n_feat, ds[1], patch_size, k)
        self.au_l1 = AggregationUnit(n_feat, ds[2], patch_size, k)

        self.conv_up = nn.Sequential(
            nn.Conv2d(n_feat * n_frames, n_feat * 4, 1),
            nn.PixelShuffle(2)
        )

    def forward(
            self, feats_l1: torch.Tensor, feats_l2: torch.Tensor, feats_l3: torch.Tensor
    ) -> torch.Tensor:
        """
        :param feats_l1: [B, C, T, H, W]
        :param feats_l2: [B, C, T, H, W]
        :param feats_l3: [B, C, T, H, W]
        :return: [B, C, 2H, 2W]
        """
        center_feat_l1 = feats_l1[:, :, self.n_frames // 2]
        center_feat_l2 = feats_l2[:, :, self.n_frames // 2]
        center_feat_l3 = feats_l3[:, :, self.n_frames // 2]

        attn_l3 = []
        for t in range(self.n_frames):
            attn = self.au_l3(center_feat_l3, feats_l3[:, :, t])
            attn_l3.append(
                F.upsample(attn, scale_factor=2, mode='bilinear', align_corners=False)
            )

        attn_l2 = []
        for t in range(self.n_frames):
            attn = self.au_l2(center_feat_l2 + attn_l3[t], feats_l2[:, :, t] + attn_l3[t])
            attn_l2.append(
                F.upsample(attn, scale_factor=2, mode='bilinear', align_corners=False)
            )

        attn_l1 = []
        for t in range(self.n_frames):
            attn_l1.append(self.au_l1(center_feat_l1 + attn_l2[t], feats_l1[:, :, t] + attn_l2[t]))

        attn_l1 = torch.cat(attn_l1, dim=1)
        attn_l1 = self.conv_up(attn_l1)
        return attn_l1

