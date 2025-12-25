import torch


def smoothness_loss(xyz, colors, k=8):
    """
    xyz:    (B, N, 3)
    colors: (B, N, 3)
    """
    B, N, _ = xyz.shape

    # pairwise distances
    dist = torch.cdist(xyz, xyz)  # (B, N, N)

    # nearest neighbors (exclude self)
    idx = dist.topk(k + 1, largest=False)[1][:, :, 1:]  # (B, N, k)

    loss = 0.0
    for b in range(B):
        neighbor_colors = colors[b][idx[b]]   # (N, k, 3)
        center_colors = colors[b].unsqueeze(1)  # (N, 1, 3)
        loss += (center_colors - neighbor_colors).abs().mean()

    return loss / B
