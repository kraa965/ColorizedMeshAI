import os
import numpy as np
import trimesh
import open3d as o3d

# Корень проекта = на уровень выше папки utils/, где лежит этот файл.
# Так пути вида "data/test/..." работают одинаково независимо от того,
# откуда запущен скрипт (из корня, из utils/, из IDE и т.д.)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def strip_colors(obj_path, out_path, neutral_gray=0.75, visualize=True):
    """
    Загружает OBJ и убирает исходную раскраску (задаёт всем вершинам
    нейтральный серый цвет). Полезно для визуального сравнения "до/после"
    с результатом нейросети без влияния исходных цветов скана.
    """
    mesh = trimesh.load(obj_path, process=False)

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("OBJ must contain exactly one mesh")

    V = mesh.vertices.astype(np.float32)
    n = len(V)

    # ---------- задаём нейтральный серый цвет всем вершинам ----------
    gray = np.full((n, 3), neutral_gray, dtype=np.float32)
    colors_uint8 = (gray * 255).astype(np.uint8)
    mesh.visual.vertex_colors = colors_uint8

    mesh.export(out_path)
    print(f"✅ Saved mesh without coloring: {out_path}")

    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(V)
        pcd.colors = o3d.utility.Vector3dVector(gray)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Mesh without coloring",
            width=900,
            height=900
        )


if __name__ == "__main__":
    obj_path = os.path.join(PROJECT_ROOT, "test/Untitled.obj")
    out_path = os.path.join(PROJECT_ROOT, "test/result_seg_uncolored_lower.obj")

    strip_colors(
        obj_path=obj_path,
        out_path=out_path,
        neutral_gray=0.75,
        visualize=True
    )