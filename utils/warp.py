import torch
import torch.nn.functional as F


def flow_warp(sample: torch.Tensor, flow: torch.Tensor):
    h, w = sample.shape[-2:]

    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(flow), torch.arange(0, w).type_as(flow))
    grid = torch.stack((grid_x, grid_y), dim=2).float().unsqueeze(0)

    grid = grid + flow.permute(0, 2, 3, 1).contiguous()
    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(w - 1, 1) - 1.0  # Normalize to [-1, 1]
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid = torch.clamp(grid, -1, 1)
    return F.grid_sample(sample, grid, mode='bilinear', align_corners=False)


def patch_align(sample: torch.Tensor, flow: torch.Tensor, patch_size: int = 8):
    """
    Rethinking Alignment in Video Super-Resolution Transformers (https://arxiv.org/pdf/2207.08494)
    :param sample:
    :param flow:
    :param patch_size:
    :return:
    """
    h, w = sample.shape[-2:]

    flow_mean = F.avg_pool2d(flow, kernel_size=patch_size, stride=patch_size)  # (B, 2, h//patch_size, w//patch_size)
    flow_mean = torch.repeat_interleave(flow_mean, patch_size, dim=-1)  # (B, 2, h//patch_size, w)
    flow_mean = torch.repeat_interleave(flow_mean, patch_size, dim=-2)  # (B, 2, h, w)

    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(flow), torch.arange(0, w).type_as(flow))
    grid = torch.stack((grid_x, grid_y), dim=2).float().unsqueeze(0)

    grid = grid + flow_mean.permute(0, 2, 3, 1).contiguous()
    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / max(w - 1, 1) - 1.0  # Normalize to [-1, 1]
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid = torch.clamp(grid, -1, 1)
    return F.grid_sample(sample, grid, mode='nearest')
