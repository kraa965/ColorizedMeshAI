import numpy as np

# общая функция нормализации координат, используется и в prepare_dataset.py, и в infer_visual.py


def normalize_vertices(V):
    """
    V: (N, 3)
    Center + uniform scale to unit sphere. Same logic used both at
    dataset preparation time and at inference time, so train/test
    normalization never drifts apart.
    """
    center = V.mean(axis=0, keepdims=True)
    Vc = V - center
    scale = np.max(np.linalg.norm(Vc, axis=1))
    Vn = Vc / (scale + 1e-8)
    return Vn, center, scale
