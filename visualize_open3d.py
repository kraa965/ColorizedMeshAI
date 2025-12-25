import numpy as np
import open3d as o3d


points = np.load("data/processed/upper/30.09.2025-Краморев А.С.-maxillary/points.npy")
colors = np.load("data/processed/upper/30.09.2025-Краморев А.С.-maxillary/colors.npy")

xyz = points[:, :3]
rgb = colors

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

o3d.visualization.draw_geometries([pcd])
