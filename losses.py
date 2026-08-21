import torch

# функция smoothness_loss, используется при обучении вместе с L1


def smoothness_loss(xyz, colors, k=8):
    """
    xyz:    (B, N, 3)
    colors: (B, N, 3)

    Encourages predicted colors to be similar to their k nearest
    geometric neighbors (Laplacian-style smoothing term).
    """
    B, N, _ = xyz.shape

    dist = torch.cdist(xyz, xyz)                        # (B, N, N)
    idx = dist.topk(k + 1, largest=False)[1][:, :, 1:]   # (B, N, k), exclude self

    batch_idx = torch.arange(B, device=xyz.device).view(B, 1, 1).expand(B, N, k)
    neighbor_colors = colors[batch_idx, idx]             # (B, N, k, 3)
    center_colors = colors.unsqueeze(2)                   # (B, N, 1, 3)

    loss = (center_colors - neighbor_colors).abs().mean()
    return loss
