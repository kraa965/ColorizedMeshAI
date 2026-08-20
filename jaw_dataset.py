import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# класс Dataset, читает подготовленные .npy из data/processed, сэмплит n_points точек на каждый скан


class JawProcessedDataset(Dataset):
    def __init__(self, root_dir, n_points=4096):
        """
        root_dir: data/processed/upper  or  data/processed/lower
        """
        self.root_dir = root_dir
        self.n_points = n_points
        self.cases = sorted(glob.glob(os.path.join(root_dir, '*')))

        assert len(self.cases) > 0, f"No cases found in {root_dir}"

    def __len__(self):
        return len(self.cases)

    def _sample(self, points, colors):
        n = len(points)
        idx = np.random.choice(n, self.n_points, replace=n < self.n_points)
        return points[idx], colors[idx]

    def __getitem__(self, idx):
        case_dir = self.cases[idx]

        points = np.load(os.path.join(case_dir, 'points.npy'))  # (N, 6)
        colors = np.load(os.path.join(case_dir, 'colors.npy'))  # (N, 3)

        points, colors = self._sample(points, colors)

        return (
            torch.from_numpy(points).float(),   # (n_points, 6)
            torch.from_numpy(colors).float()    # (n_points, 3)
        )
