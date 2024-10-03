import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from models.FRVSR import FNet  # used for flow estimation
from layers.upsample import AdHocSPConv
from utils.warp import flow_warp

from cv2 import getGaussianKernel


class LaplacianEnhancementModule(nn.Module):
    def __init__(self, alpha: float = 1, sigma: float = 1.5):
        super(LaplacianEnhancementModule, self).__init__()
        self.alpha = alpha
        self.sigma = sigma

        # kernel size is 3 not described in the paper
        gaussian_1d = getGaussianKernel(3, self.sigma)
        gaussian_2d = gaussian_1d @ gaussian_1d.T
        gaussian_2d = torch.from_numpy(gaussian_2d).float().unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1)  # C_out, C_in, H, W
        self.register_buffer('gaussian_2d', gaussian_2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * (x - F.conv2d(x, self.gaussian_2d, padding=1, groups=3))


class DetailAwareFlowEstimation(nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet = FNet()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, 3, 1)
        )

    def forward(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flow_lr = self.fnet(xs)
        flow_hr = self.deconv(flow_lr) + F.interpolate(
            flow_lr, scale_factor=4, mode='bilinear', align_corners=True
        ) * 4
        return flow_lr, flow_hr


class COMISR(nn.Module):
    def __init__(self):
        super(COMISR, self).__init__()
        self.le = LaplacianEnhancementModule()
        self.dafe = DetailAwareFlowEstimation()
        self.reconstruction = nn.Sequential(  # reconstruction network is not described in the paper and original code is tfv1.
            nn.Conv2d(3 + (3 * 16), 64, 3, 1, 1),  # 3 * 16 = space2depth of warped hr_tm1
            AdHocSPConv(64, 4, 3)
        )

    def forward(self, lr: torch.Tensor, lr_tm1: torch.Tensor, hr_tm1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param lr: [B, 3, H, W]
        :param lr_tm1: [B, 3, H, W]
        :param hr_tm1: [B, 3, 4H, 4W]
        :return: [B, 3, 4H, 4W], [B, 3, H, W]
        """
        flow_lr, flow_hr = self.dafe(torch.cat([lr, lr_tm1], dim=1))

        lr_tm1_warped = flow_warp(lr_tm1, flow_lr)
        hr_tm1_warped = flow_warp(hr_tm1, flow_hr)

        hr_tm1_warped = self.le(hr_tm1_warped)
        hr_tm1_warped = F.pixel_unshuffle(hr_tm1_warped, 4)

        return self.reconstruction(torch.cat([lr, hr_tm1_warped], dim=1)), lr_tm1_warped
