import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import deform_conv2d


# Without Mask
class DeformableConv2d(nn.Module):
    def __init__(self, c_base: int, deformable_groups: int):
        super().__init__()
        self.register_parameter(
            "weight", nn.Parameter(torch.randn(c_base, c_base, 3, 3) * 0.02)
        )
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(c_base))
        )
        self.offset_mask_conv = nn.Conv2d(
            c_base, 2 * deformable_groups * 3 * 3, 3, 1, 1  # offset * groups * kh * kw
        )
        self.init_weights()

    def init_weights(self):
        self.offset_mask_conv.weight.data.zero_()
        self.offset_mask_conv.bias.data.zero_()

    def forward(self, feat: torch.Tensor, reference: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param feat: [B, C, H, W]
        :param reference: [B, C, H, W] or None
        :return: [B, C, H, W]
        """
        offset = self.offset_mask_conv(reference) if reference is not None else self.offset_mask_conv(feat)
        return deform_conv2d(feat, offset, self.weight, self.bias, padding=(1, 1))

    def extra_repr(self):
        return f"(weight): {self.weight.shape} \n(bias): {self.bias.shape}"


# With Mask
class DeformableConv2dPack(nn.Module):
    def __init__(self, c_base: int, deformable_groups: int):
        super().__init__()
        self.register_parameter(
            "weight", nn.Parameter(torch.randn(c_base, c_base, 3, 3) * 0.02)
        )
        self.register_parameter(
            'bias', nn.Parameter(torch.zeros(c_base))
        )
        self.offset_mask_conv = nn.Conv2d(
            c_base, 3 * deformable_groups * 3 * 3, 3, 1, 1  # (offset + mask) * groups * kh * kw
        )
        self.init_weights()

    def init_weights(self):
        self.offset_mask_conv.weight.data.zero_()
        self.offset_mask_conv.bias.data.zero_()

    def forward(self, x: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor of shape (B, C, H, W)
        :param feat: Tensor of shape (B, C, H, W)
        :return: Tensor of shape (B, C, H, W)
        """
        out = self.offset_mask_conv(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(x, offset, self.weight, self.bias, padding=(1, 1), mask=mask)

    def extra_repr(self):
        return f"(weight): {self.weight.shape} \n(bias): {self.bias.shape}"
