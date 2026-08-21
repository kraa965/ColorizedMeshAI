import numpy as np
import trimesh
import open3d as o3d
from skimage import color
from sklearn.neighbors import NearestNeighbors


# ---------- utils ----------

def visualize(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def knn_smooth(labels, points, k=20):
    """
    Majority vote smoothing
    """
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, idx = nbrs.kneighbors(points)

    new_labels = np.zeros_like(labels)
    for i in range(len(labels)):
        votes = labels[idx[i]]
        new_labels[i] = np.round(votes.mean())

    return new_labels


# ---------- main ----------

def segment_obj(obj_path):
    mesh = trimesh.load(obj_path, process=False)

    V = mesh.vertices.astype(np.float32)

    # RGB in [0,1]
    C = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0

    # ---------- LAB ----------
    lab = color.rgb2lab(C)
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]

    # ---------- GEOMETRY ----------
    normals = mesh.vertex_normals
    curvature = np.linalg.norm(normals - normals.mean(axis=0), axis=1)

    # ---------- HEURISTIC RULE ----------
    tooth_mask = (
        (A < 15) &        # не красный
        (L > 60) &        # светлый
        (curvature > 0.02)  # выпуклый
    )

    tooth_mask = tooth_mask.astype(np.int32)

    # ---------- SPATIAL SMOOTHING ----------
    tooth_mask = knn_smooth(tooth_mask, V, k=30)

    # ---------- VISUALIZE ----------
    vis_color = np.zeros_like(C)
    vis_color[tooth_mask == 1] = [0.95, 0.95, 0.95]  # зубы
    vis_color[tooth_mask == 0] = [1.0, 0.4, 0.5]    # десна

    visualize(V, vis_color)

    return tooth_mask


if __name__ == "__main__":
    obj_path = "../test/upper1.obj"
    segment_obj(obj_path)
