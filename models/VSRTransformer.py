import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.SPyNet import SPyNet  # For Flow estimation
from layers.positional_encoding import PositionalEncodingPermute3D
from utils.warp import flow_warp


class LayerNorm3d(nn.Module):
    def __init__(self, n_feats: int):
        super().__init__()
        self.norm = nn.LayerNorm(n_feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        else:
            x = x.permute(0, 2, 3, 4, 1)
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3)
        return x


class SpatioTemporalConvolutionalSA(nn.Module):
    def __init__(self, n_feats: int, patch_size: int, n_heads: int = 1):
        super().__init__()
        self.n_feats = n_feats
        self.patch_size = patch_size
        self.n_heads = n_heads
        self.scale = n_feats ** -0.5

        self.to_q = nn.Conv2d(n_feats, n_feats, 3, 1, 1, groups=n_feats)
        self.to_k = nn.Conv2d(n_feats, n_feats, 3, 1, 1, groups=n_feats)
        self.to_v = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        self.to_out = nn.Conv3d(n_feats, n_feats, (1, 3, 3), 1, (0, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, C, T, H, W]
        :return: [B, C, T, H, W]
        """
        b, c, t, h, w = x.shape
        h_ = h // self.patch_size
        w_ = w // self.patch_size

        feat = rearrange(x, 'b c t h w -> (b t) c h w')
        q = self.to_q(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)

        # similar with patch-merging but to channel
        # k1 in the paper
        q = rearrange(
            q, '(b t) (heads c) (h ph) (w pw) -> b heads (t ph pw) (h w c)',
            heads=self.n_heads, ph=self.patch_size, pw=self.patch_size, t=t
        )
        k = rearrange(
            k, '(b t) (heads c) (h ph) (w pw) -> b heads (t ph pw) (h w c)',
            heads=self.n_heads, ph=self.patch_size, pw=self.patch_size, t=t
        )
        v = rearrange(
            v, '(b t) (heads c) (h ph) (w pw) -> b heads (t ph pw) (h w c)',
            heads=self.n_heads, ph=self.patch_size, pw=self.patch_size, t=t
        )

        attn = q @ k.transpose(-2, -1)
        attn = F.softmax(attn, dim=-1) * self.scale
        out = attn @ v

        # k2 in the paper
        out = rearrange(
            out, 'b heads (t ph pw) (h w c) -> b (heads c) t (h ph) (w pw)',
            heads=self.n_heads, ph=self.patch_size, pw=self.patch_size, h=h_, w=w_
        )

        out = self.to_out(out)
        return out


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


class BidirectionalOpticalFlowFFN(nn.Module):
    def __init__(self, n_feats: int, n_blocks: int):
        super().__init__()
        self.backward_resblocks = ResBlocksInputConv(n_feats + 3, n_feats, n_blocks)
        self.forward_resblocks = ResBlocksInputConv(n_feats + 3, n_feats, n_blocks)
        self.fusion = nn.Conv3d(n_feats * 2, n_feats, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor, lrs: torch.Tensor, flows: tuple[torch.Tensor]) -> torch.Tensor:
        """
        :param x:  [B, C, T, H, W]
        :param lrs:  [BxT, 3, H, W]
        :param flows:  [BxT, 2, H, W], [BxT, 2, H, W]
        :return:
        """
        b, c, t, h, w = x.shape

        x_backward = rearrange(
            torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2), 'b c t h w -> (b t) c h w'
        )  # [BxT, C, H, W]
        x_backward = flow_warp(x_backward, flows[1])
        x_backward = torch.cat([lrs, x_backward], dim=1)
        x_backward = self.backward_resblocks(x_backward)
        x_backward = rearrange(x_backward, '(b t) c h w -> b c t h w', b=b, t=t)

        x_forward = rearrange(
            torch.cat([x[:, :, :-1], x[:, :, 0:1]], dim=2), 'b c t h w -> (b t) c h w'
        )
        x_forward = flow_warp(x_forward, flows[0])
        x_forward = torch.cat([lrs, x_forward], dim=1)
        x_forward = self.forward_resblocks(x_forward)
        x_forward = rearrange(x_forward, '(b t) c h w -> b c t h w', b=b, t=t)

        out = torch.cat([x_backward, x_forward], dim=1)
        out = self.lrelu(self.fusion(out))
        return out


class BidirectionalSPyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spynet = SPyNet()

    def forward(self, lrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param lrs: [B, 3, T, H, W]
        :return: [B, T, 2, H, W], [B, T, 2, H, W]
        """
        b, c, t, h, w = lrs.shape
        t_pad = t + 1
        lrs_pad_f = rearrange(
            F.pad(lrs, (0, 0, 0, 0, 1, 0), mode='replicate'),
            'b c t h w -> (b t) c h w'
        )
        lrs_pad_b = rearrange(
            F.pad(lrs, (0, 0, 0, 0, 0, 1), mode='replicate'),
            'b c t h w -> (b t) c h w'
        )

        flow_f = self.spynet(lrs_pad_f, lrs_pad_b)
        flow_b = self.spynet(lrs_pad_b, lrs_pad_f)

        flow_f = rearrange(flow_f, '(b t) c h w -> b t c h w', b=b, t=t_pad)[:, 1:]
        flow_b = rearrange(flow_b, '(b t) c h w -> b t c h w', b=b, t=t_pad)[:, :-1]

        return flow_f, flow_b


class Transformer(nn.Module):
    def __init__(
            self, n_blocks: int, n_feats: int,
            patch_size: int, n_heads: int, n_resblocks: int,
    ):
        super().__init__()

        self.sa = nn.ModuleList([
            SpatioTemporalConvolutionalSA(n_feats, patch_size, n_heads) for _ in range(n_blocks)
        ])
        self.sa_postnorm = nn.ModuleList([
            LayerNorm3d(n_feats) for _ in range(n_blocks)
        ])
        self.ffn = nn.ModuleList([
            BidirectionalOpticalFlowFFN(n_feats, n_resblocks) for _ in range(n_blocks)
        ])
        self.ffn_postnorm = nn.ModuleList([
            LayerNorm3d(n_feats) for _ in range(n_blocks)
        ])

    def forward(
            self, x: torch.Tensor, lrs: torch.Tensor, flows: tuple[torch.Tensor]
    ) -> torch.Tensor:
        """
        :param x: [B, C, T, H, W]
        :param lrs:  [BxT, 3, H, W]
        :param flows:  [BxT, 2, H, W], [BxT, 2, H, W]
        :return: [B, C, T, H, W]
        """
        for sa, sa_postnorm, ffn, ffn_postnorm in zip(
                self.sa, self.sa_postnorm, self.ffn, self.ffn_postnorm
        ):
            x = sa_postnorm(sa(x) + x)
            x = ffn_postnorm(ffn(x, lrs, flows) + x)
        return x


class VSRTransformer(nn.Module):
    def __init__(
            self, n_feats: int, n_proj_blocks: int, n_frames: int,
            n_tx_blocks: int, n_resblocks: int, patch_size: int, n_heads: int,
            n_recon_blocks: int,
    ):
        super().__init__()
        self.flow = BidirectionalSPyNet()

        self.feat_proj = nn.Conv2d(3, n_feats, 3, 1, 1)
        self.feat_proj_refine = nn.Sequential(*[
            ResBlock(n_feats) for _ in range(n_proj_blocks)
        ])
        self.pe = PositionalEncodingPermute3D(n_frames)

        self.tx = Transformer(n_tx_blocks, n_feats, patch_size, n_heads, n_resblocks)

        self.recon = nn.Sequential(
            *[ResBlock(n_feats) for _ in range(n_recon_blocks)],
            nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(n_feats, 3, 3, 1, 1)
        )

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        """
        :param lrs:  [B, 3, T, H, W]
        :return:  [B, 3, T, 4H, 4W]
        """
        b, _, t, h, w = lrs.shape

        flows = self.flow(lrs)
        flows = [
            rearrange(flows[0], 'b t c h w -> (b t) c h w'),
            rearrange(flows[1], 'b t c h w -> (b t) c h w')
        ]

        lrs = rearrange(lrs, 'b c t h w -> (b t) c h w')
        feats = self.feat_proj(lrs)
        feats = rearrange(
            self.feat_proj_refine(feats), '(b t) c h w -> b c t h w', b=b
        )
        feats = feats + self.pe(
            feats.permute(0, 2, 1, 3, 4).contiguous()
        ).permute(0, 2, 1, 3, 4).contiguous()

        feats = self.tx(feats, lrs, flows)

        out = self.recon(rearrange(feats, 'b c t h w -> (b t) c h w'))
        out = rearrange(out, '(b t) c h w -> b c t h w', b=b)
        lrs_up = F.interpolate(
            lrs, scale_factor=4, mode='bilinear', align_corners=False
        )
        lrs_up = rearrange(lrs_up, '(b t) c h w -> b c t h w', b=b)
        out = out + lrs_up

        return out
