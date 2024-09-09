import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from layers.deformable_conv_v2 import DeformableConv2d


class ResBlock(nn.Sequential):
    def __init__(self, n_feats: int, is_3d: bool = True):
        layers = [
            nn.Conv2d(n_feats, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        ] if not is_3d else [
            nn.Conv3d(n_feats, n_feats, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_feats, n_feats, (1, 3, 3), (1, 1, 1), (0, 1, 1))
        ]
        super().__init__(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)


class FeatureExtraction(nn.Module):
    def __init__(self, n_feats: int, k1: int = 5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(3, n_feats, (1, 3, 3), (1, 1, 1), (0, 1, 1)),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(*[ResBlock(n_feats) for _ in range(k1)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: [B, 3, T, H, W]
        :return: [B, C, T, H, W]
        """
        t = x.size(2)
        x = self.proj(x)
        x_t = x[:, :, t // 2, :, :]
        x = self.res_blocks(x)
        return x, x_t


class DeformableAlignment(nn.Module):
    def __init__(self, n_feats: int, n_frames: int, deformable_groups: int = 8):
        super().__init__()
        self.n_feats = n_feats
        self.n_frames = n_frames

        self.bottleneck = nn.Conv2d(n_feats * 2, n_feats, 3, 1, 1)

        # 2 deformable conv before f_dc and 1 after
        self.bef_1 = DeformableConv2d(n_feats, deformable_groups=deformable_groups)
        self.bef_2 = DeformableConv2d(n_feats, deformable_groups=deformable_groups)

        self.f_dc = DeformableConv2d(n_feats, deformable_groups=deformable_groups)

        self.aft_1 = DeformableConv2d(n_feats, deformable_groups=deformable_groups)

        self.recon = nn.Conv2d(n_feats, 3, 3, 1, 1)

    def forward(self, t_frame: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """
        :param t_frame: [B, C, H, W]
        :param frames: [B, C, T, H, W]
        :return: [B, 3, T, H, W]
        """
        res = []
        for t in range(self.n_frames):
            refer = frames[:, :, t, :, :]

            feat = torch.cat([refer, t_frame], dim=1)
            feat = self.bottleneck(feat)
            feat = self.bef_1(feat)
            feat = self.bef_2(feat)
            feat = self.f_dc(refer, feat)
            feat = self.aft_1(feat)
            recon = self.recon(feat)
            res.append(recon)

        return torch.stack(res, dim=2)


class Upsampler(nn.Sequential):
    def __init__(self, n_feats: int, scale: int):
        layers = [
            nn.Conv2d(n_feats, n_feats * scale ** 2, 3, 1, 1),
            nn.PixelShuffle(scale)
        ] if scale < 4 else [
            nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(n_feats, n_feats * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        ]
        layers.append(nn.Conv2d(n_feats, 3, 3, 1, 1))
        super().__init__(*layers)


class Reconstruction(nn.Module):
    def __init__(self, n_feats: int, n_frames: int, k2: int = 10, scale: int = 4):
        super().__init__()
        self.n_feats = n_feats
        self.n_frames = n_frames
        self.scale = scale

        self.temporal_fusion = nn.Sequential(
            nn.Conv2d(3 * n_frames, n_feats, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.nonlinear_mapping = nn.Sequential(*[
            ResBlock(n_feats, is_3d=False) for _ in range(k2)
        ])
        self.hr_recon = Upsampler(n_feats, scale)

    def forward(self, frames_aligned: torch.Tensor) -> torch.Tensor:
        """
        :param frames_aligned: [B, 3, T, H, W]
        :return: [B, 3, rH, rW]
        """
        frames = rearrange(frames_aligned, 'b c t h w -> b (c t) h w')
        frames = self.temporal_fusion(frames)
        frames = self.nonlinear_mapping(frames)
        frames = self.hr_recon(frames)
        return frames


class TDAN(nn.Module):
    def __init__(self, n_feats: int, n_frames: int, k1: int = 5, k2: int = 10, scale: int = 4):
        super().__init__()
        self.feature_extraction = FeatureExtraction(n_feats, k1)
        self.deformable_alignment = DeformableAlignment(n_feats, n_frames)
        self.reconstruction = Reconstruction(n_feats, n_frames, k2, scale)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        :param frames: [B, 3, T, H, W]
        :return: [B, 3, rH, rW]
        """
        frames, t_frame = self.feature_extraction(frames)
        frames_aligned = self.deformable_alignment(t_frame, frames)
        frames_hr = self.reconstruction(frames_aligned)
        return frames_hr
