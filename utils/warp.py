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
