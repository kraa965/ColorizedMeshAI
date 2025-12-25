import numpy as np
import torch
import trimesh
import open3d as o3d

from model_pointnetpp import PointNetPPColor


# ---------- utils ----------

def normalize_vertices(V):
    """
    V: (N, 3)
    """
    center = V.mean(axis=0, keepdims=True)
    Vc = V - center
    scale = np.max(np.linalg.norm(Vc, axis=1))
    Vn = Vc / (scale + 1e-8)
    return Vn, center, scale


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


# ---------- inference ----------

@torch.no_grad()
def infer(
    obj_path,
    model_path,
    out_path,
    n_points=4096,
    device="cuda",
    visualize=True
):
    # ---------- load mesh (ЕДИНЫЙ ИСТОЧНИК) ----------
    mesh = trimesh.load(obj_path, process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("OBJ must contain exactly one mesh")

    V = mesh.vertices.astype(np.float32)           # (N, 3)
    N = mesh.vertex_normals.astype(np.float32)     # (N, 3)

    assert V.shape == N.shape, "Vertices and normals must match"

    # ---------- normalize ----------
    Vn, center, scale = normalize_vertices(V)

    # ---------- model input ----------
    X = np.concatenate([Vn, N], axis=1)  # (N, 6)

    # ---------- load model ----------
    model = PointNetPPColor().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ---------- inference (chunked) ----------
    preds = np.zeros((len(X), 3), dtype=np.float32)
    bs = n_points

    for i in range(0, len(X), bs):
        chunk = X[i:i + bs]

        if len(chunk) < bs:
            pad = bs - len(chunk)
            chunk = np.pad(chunk, ((0, pad), (0, 0)), mode="edge")

        chunk = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, bs, 6)
        pred = model(chunk)[0].cpu().numpy()                     # (bs, 3)

        preds[i:i + min(bs, len(X) - i)] = pred[:len(X) - i]

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
    obj_path = "data/test/upper1.obj"
    model_path = "weights/best.pth"
    out_path = "result_colored_upper.obj"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    infer(
        obj_path=obj_path,
        model_path=model_path,
        out_path=out_path,
        n_points=4096,
        device=device,
        visualize=True
    )
