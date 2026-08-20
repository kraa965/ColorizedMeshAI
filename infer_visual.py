import numpy as np
import torch
import trimesh
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from model_pointnetpp import PointNetPPColor
from geometry_utils import normalize_vertices


# загружает обученные веса, красит новый скан, сохраняет результат и открывает 3D-визуализацию

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
    over k spatial nearest neighbors. NOTE: this blends across the
    tooth/gum boundary indiscriminately, so it can wash out contrast
    the model itself predicted correctly. Off by default (smooth_k=0)
    so infer() returns the model's raw output for honest comparison
    between checkpoints.
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
    smooth_k=0
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

    # ---------- optional postprocessing smoothing (OFF by default) ----------
    # Раньше здесь всегда применялось сглаживание (smooth_k=12), из-за чего
    # в .obj попадал не чистый выход модели, а выход + фильтр поверх, что
    # мешало честно сравнивать чекпоинты в Blender. По умолчанию smooth_k=0 -
    # постобработка выключена, экспортируется прямое предсказание сети.
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


# ---------- entry point ----------

if __name__ == "__main__":
    obj_path = "data/test/Untitled.obj"
    model_path = "weights/best_epoch0497_loss0.056206.pth"
    out_path = "result_colored_upper.obj"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    infer(
        obj_path=obj_path,
        model_path=model_path,
        out_path=out_path,
        n_points=4096,
        k_neighbors=16,
        device=device,
        visualize=True,
        smooth_k=0   # чистый выход модели, без постобработки
    )