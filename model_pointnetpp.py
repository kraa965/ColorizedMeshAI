import torch
import torch.nn as nn
import torch.nn.functional as F


# сама архитектура сети (EdgeConv/DGCNN), предсказывает цвет по геометрии + метке сегмента


def knn_idx(xyz, k):
    """
    xyz: (B, N, 3)
    returns idx: (B, N, k) indices of k nearest neighbors (excluding self)
    """
    dist = torch.cdist(xyz, xyz)                      # (B, N, N)
    idx = dist.topk(k + 1, largest=False)[1][:, :, 1:]  # drop self (dist=0)
    return idx


def gather_neighbors(feat, idx):
    """
    feat: (B, N, C)
    idx:  (B, N, k)
    returns: (B, N, k, C)
    """
    B, N, C = feat.shape
    k = idx.shape[-1]
    batch_idx = torch.arange(B, device=feat.device).view(B, 1, 1).expand(B, N, k)
    out = feat[batch_idx, idx]  # (B, N, k, C)
    return out


class EdgeConv(nn.Module):
    """
    DGCNN-style edge convolution: for each point, aggregate over its k
    nearest neighbors using [center_feat, neighbor_feat - center_feat],
    then max-pool over neighbors. This is what gives the network actual
    spatial context, unlike a plain per-point MLP.
    """

    def __init__(self, in_ch, out_ch, k=16):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, xyz, feat):
        """
        xyz:  (B, N, 3)  coordinates used to find neighbors
        feat: (B, N, C)  features to aggregate
        returns: (B, N, out_ch)
        """
        idx = knn_idx(xyz, self.k)                 # (B, N, k)
        neighbor_feat = gather_neighbors(feat, idx)  # (B, N, k, C)
        center_feat = feat.unsqueeze(2).expand(-1, -1, self.k, -1)  # (B, N, k, C)

        edge_feat = torch.cat([center_feat, neighbor_feat - center_feat], dim=-1)  # (B, N, k, 2C)
        edge_feat = edge_feat.permute(0, 3, 1, 2)   # (B, 2C, N, k)

        out = self.mlp(edge_feat)                   # (B, out_ch, N, k)
        out = out.max(dim=-1)[0]                     # (B, out_ch, N)
        return out.permute(0, 2, 1)                  # (B, N, out_ch)


class PointNetPPColor(nn.Module):
    """
    DGCNN-style сеть. Вход - 8 признаков на точку:
        xyz (3) + normals (3) + segment_label (1: 1=зуб, 0=десна) + label_mask (1)

    label_mask=1 означает, что segment_label - реальная метка из точной
    геометрической сегментации (разделённые меши зуб/десна). label_mask=0
    означает, что сегментация недоступна (единый меш без групп) - тогда
    segment_label всегда 0, и сеть ориентируется только на геометрию.
    Обучение со случайным "выключением" маски (см. jaw_dataset.py) учит
    сеть работать в обоих режимах на одном датасете.
    """

    def __init__(self, k=16, in_ch=8):
        super().__init__()
        self.k = k

        self.ec1 = EdgeConv(in_ch, 64, k=k)
        self.ec2 = EdgeConv(64, 128, k=k)
        self.ec3 = EdgeConv(128, 128, k=k)

        self.head = nn.Sequential(
            nn.Conv1d(64 + 128 + 128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: (B, N, 8)  -> XYZ + normals + segment label + label mask
        """
        assert x.dim() == 3 and x.size(2) == 8, f"Expected (B,N,8), got {x.shape}"

        xyz = x[:, :, :3]

        f1 = self.ec1(xyz, x)        # (B, N, 64)  neighbors found via input xyz
        f2 = self.ec2(xyz, f1)       # (B, N, 128)
        f3 = self.ec3(xyz, f2)       # (B, N, 128)

        feat = torch.cat([f1, f2, f3], dim=-1)  # (B, N, 320)
        feat = feat.permute(0, 2, 1)            # (B, 320, N)

        out = self.head(feat)                   # (B, 3, N)
        return out.permute(0, 2, 1)              # (B, N, 3)
