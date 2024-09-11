import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import math


class ConvNets(nn.Sequential):
    def __init__(self):
        super().__init(
            nn.Conv2d(8, 32, 7, 1, 3),  # Ref(RGB) + Sup(RGB) + Flow(UV)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, 7, 1, 3),  # Flow(UV)
        )


class SPyNet(nn.Module):
    def __init__(self, n_levels=6):
        super().__init__()
        # 5 for FlyingChairs, 6 for Sintel
        self.n_levels = n_levels
        self.Gs = nn.ModuleList([ConvNets() for _ in range(n_levels)])

    @staticmethod
    def d(sample):
        return F.avg_pool2d(sample, 2, 2)

    @staticmethod
    def u(sample):
        return F.interpolate(
            sample, scale_factor=2, mode='bilinear', align_corners=False
        ) * 2  # *2 for enhance magnitude

    @staticmethod
    def w(sample, flow):
        """
        :param sample: B, 3, H, W
        :param flow: B, 2, H, W
        :return: B, 3, H, W
        """
        h, w = flow.size(2), flow.size(3)
        flow[:, 0] = flow[:, 0] / (w / 2)  # we use align_corners=False
        flow[:, 1] = flow[:, 1] / (h / 2)

        return F.grid_sample(sample, flow.permute(0, 2, 3, 1), mode='bilinear', align_corners=False)

    def forward(self, ref, sup):
        base_size = 2 ** (self.n_levels - 1)
        h, w = ref.size(2), ref.size(3)
        h_to = math.floor(math.ceil(h / base_size) * base_size)
        w_to = math.floor(math.ceil(w / base_size) * base_size)

        ref = F.interpolate(ref, size=(h_to, w_to), mode='bilinear', align_corners=False)
        sup = F.interpolate(sup, size=(h_to, w_to), mode='bilinear', align_corners=False)

        # Build pyramid
        ref_pyramid = [ref]
        sup_pyramid = [sup]

        for i in range(self.n_levels - 1):
            ref_pyramid.append(self.d(ref_pyramid[-1]))
            sup_pyramid.append(self.d(sup_pyramid[-1]))

        ref_pyramid = ref_pyramid[::-1]
        sup_pyramid = sup_pyramid[::-1]

        flow = torch.zeros(ref.size(0), 2, ref_pyramid[0].size(2), ref_pyramid[0].size(3)).to(ref.device)

        i = 0
        for ref_k, sup_k, g_k in zip(ref_pyramid, sup_pyramid, self.Gs):
            if i > 0:
                sup_k = self.w(sup_k, flow)
            flow = g_k(torch.cat([ref_k, sup_k, flow], dim=1)) + flow
            flow = self.u(flow)

            i += 1

        flow = F.interpolate(flow, size=(h, w), mode='bilinear', align_corners=False)
        flow[:, 0] = flow[:, 0] * w / w_to
        flow[:, 1] = flow[:, 1] * h / h_to
        return flow
