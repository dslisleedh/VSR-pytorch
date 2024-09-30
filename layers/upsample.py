import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class AdHocSPConv(nn.Sequential):
    def __init__(
            self, c_base: int, scale: int, n_colors: int = 3
    ):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(c_base, 4 * c_base, 3, padding=1))
                m.append(nn.PixelShuffle(2))
                m.append(nn.ReLU())
        elif scale == 3:
            m.append(nn.Conv2d(c_base, 9 * c_base, 3, padding=1))
            m.append(nn.PixelShuffle(3))
            m.append(nn.ReLU())
        else:
            raise NotImplementedError
        m += [nn.Conv2d(c_base, n_colors, 3, padding=1)]
        super(AdHocSPConv, self).__init__(*m)
