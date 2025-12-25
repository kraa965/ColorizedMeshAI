import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetPPColor(nn.Module):
    def __init__(self):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU()
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, N, 6)  -> XYZ + normals
        """
        assert x.dim() == 3 and x.size(2) == 6, f"Expected (B,N,6), got {x.shape}"

        x = x.permute(0, 2, 1)   # (B, 6, N)
        x = self.mlp1(x)         # (B, 128, N)
        x = self.mlp2(x)         # (B, 128, N)
        x = self.out(x)          # (B, 3, N)

        return x.permute(0, 2, 1)  # (B, N, 3)
