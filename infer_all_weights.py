import os
import glob
import numpy as np
import torch
import trimesh
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from model_pointnetpp import PointNetPPColor
from geometry_utils import normalize_vertices


# загружает обученные веса, красит новый скан, сохраняет результат.
# Может прогнать один файл весов или все веса из папки weights/ разом,
# сохраняя каждый результат в отдельный файл в out_dir.

# ---------- utils ----------

def visualize_open3d(V, C):
    """
    V: (N, 3)
    C: (N, 3) RGB in [0,1]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(V)
    pcd.colors = o3d.utility.Vector3dVector(C)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Predicted Vertex Colors",
        width=900,
        height=900
    )


def knn_smooth_colors(V, colors, k=12):
    """
    Postprocessing smoothing of predicted per-vertex colors, averaging
    over k spatial nearest neighbors. Helps clean up residual per-point
    noise that the model still leaves at chunk boundaries.
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(V)
    _, idx = nbrs.kneighbors(V)
    return colors[idx].mean(axis=1)


# ---------- inference ----------

@torch.no_grad()
def infer(
    obj_path,
    model_path,
    out_path,
    n_points=4096,
    k_neighbors=16,
    device="cuda",
    visualize=True,
    smooth_k=12
):
    # ---------- load mesh (единый источник) ----------
    mesh = trimesh.load(obj_path, process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("OBJ must contain exactly one mesh")

    V = mesh.vertices.astype(np.float32)           # (N, 3)
    N = mesh.vertex_normals.astype(np.float32)     # (N, 3)

    assert V.shape == N.shape, "Vertices and normals must match"

    # ---------- normalize (same function as prepare_dataset.py) ----------
    Vn, center, scale = normalize_vertices(V)

    # ---------- model input ----------
    X = np.concatenate([Vn, N], axis=1)  # (N, 6)

    # ---------- load model ----------
    model = PointNetPPColor(k=k_neighbors).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ---------- inference (chunked) ----------
    # NOTE: the model aggregates neighbors via kNN *within each chunk*,
    # so chunk boundaries will always be a bit weaker than the interior.
    # Larger n_points per chunk (or overlapping chunks) reduces this.
    preds = np.zeros((len(X), 3), dtype=np.float32)
    bs = n_points

    for i in range(0, len(X), bs):
        chunk = X[i:i + bs]

        if len(chunk) < bs:
            pad = bs - len(chunk)
            chunk = np.pad(chunk, ((0, pad), (0, 0)), mode="edge")

        chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, bs, 6)
        pred = model(chunk_t)[0].cpu().numpy()                     # (bs, 3)

        preds[i:i + min(bs, len(X) - i)] = pred[:len(X) - i]

    preds = np.clip(preds, 0.0, 1.0)

    # ---------- spatial smoothing across full mesh (fixes chunk seams too) ----------
    if smooth_k and smooth_k > 1:
        preds = knn_smooth_colors(V, preds, k=smooth_k)
        preds = np.clip(preds, 0.0, 1.0)

    # ---------- assign colors & export ----------
    colors_uint8 = (preds * 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors_uint8
    mesh.export(out_path)

    print(f"✅ Saved colored mesh: {out_path}")

    # ---------- visualize ----------
    if visualize:
        visualize_open3d(V, preds)


def infer_all_weights(
    obj_path,
    weights_dir,
    out_dir,
    n_points=4096,
    k_neighbors=16,
    device="cuda",
    smooth_k=12,
    pattern="*.pth"
):
    """
    Прогоняет один и тот же obj_path через КАЖДЫЙ файл весов из weights_dir
    и сохраняет результат каждого прогона отдельным файлом в out_dir.
    Имя выходного файла = имя файла весов (без расширения) + .obj.
    Визуализация не открывается — только сохранение на диск.
    """
    os.makedirs(out_dir, exist_ok=True)

    weight_paths = sorted(glob.glob(os.path.join(weights_dir, pattern)))

    if not weight_paths:
        raise RuntimeError(f"No weight files found in {weights_dir} matching {pattern}")

    print(f"Найдено {len(weight_paths)} файлов весов в {weights_dir}")

    for i, model_path in enumerate(weight_paths, start=1):
        weight_name = os.path.splitext(os.path.basename(model_path))[0]
        out_path = os.path.join(out_dir, f"{weight_name}.obj")

        print(f"[{i}/{len(weight_paths)}] {os.path.basename(model_path)} -> {out_path}")

        infer(
            obj_path=obj_path,
            model_path=model_path,
            out_path=out_path,
            n_points=n_points,
            k_neighbors=k_neighbors,
            device=device,
            visualize=False,   # визуализация отключена при массовом прогоне
            smooth_k=smooth_k
        )

    print(f"\n✅ Готово: {len(weight_paths)} результатов сохранено в {out_dir}")


# ---------- entry point ----------

if __name__ == "__main__":
    obj_path = "data/test/result_uncolored_upper.obj"
    weights_dir = "weights"
    out_dir = "data/test/results_by_weights"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    infer_all_weights(
        obj_path=obj_path,
        weights_dir=weights_dir,
        out_dir=out_dir,
        n_points=4096,
        k_neighbors=16,
        device=device,
        smooth_k=12,
        pattern="*.pth"   # можно сузить, например "best_epoch*.pth", чтобы пропустить best.pth/last.pth
    )