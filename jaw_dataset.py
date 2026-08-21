import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# класс Dataset, читает подготовленные .npy из data/processed, сэмплит n_points точек на каждый скан


class JawProcessedDataset(Dataset):
    def __init__(self, root_dir, n_points=4096, label_dropout_prob=0.3):
        """
        root_dir: data/processed/upper  or  data/processed/lower
        label_dropout_prob: вероятность (0..1) того, что для конкретного
            скана метка сегмента (зуб/десна) будет "выключена" целиком -
            эмулирует несегментированный вход (единый меш без групп),
            с которым модель тоже должна уметь работать на инференсе.
            0.0 - метка есть всегда (только сегментированный режим).
            1.0 - метка отключена всегда (только несегментированный режим).
        """
        self.root_dir = root_dir
        self.n_points = n_points
        self.label_dropout_prob = label_dropout_prob
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

        # points.npy: (N, 8) = xyz(3) + normals(3) + label(1) + mask(1)
        points = np.load(os.path.join(case_dir, 'points.npy'))
        colors = np.load(os.path.join(case_dir, 'colors.npy'))  # (N, 3)

        points, colors = self._sample(points, colors)

        # ---------- аугментация: эмуляция несегментированного скана ----------
        # Решение принимается на весь скан целиком (не по отдельным точкам) -
        # так честнее имитирует реальный инференс, где сегментация либо есть
        # для всего скана, либо её нет вообще.
        if np.random.rand() < self.label_dropout_prob:
            points = points.copy()
            points[:, 6] = 0.0   # label -> 0 (нейтральное значение)
            points[:, 7] = 0.0   # mask -> 0 (сигнал "метка неизвестна")

        return (
            torch.from_numpy(points).float(),   # (n_points, 8)
            torch.from_numpy(colors).float()    # (n_points, 3)
        )
