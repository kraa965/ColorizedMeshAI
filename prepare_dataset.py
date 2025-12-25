import os
import numpy as np
import trimesh
from tqdm import tqdm


RAW_ROOT = "data/raw"
OUT_ROOT = "data/processed"


def load_obj_with_vertex_color(path):
    V, C = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 7:
                    x, y, z = map(float, parts[1:4])
                    r, g, b = map(float, parts[4:7])
                    V.append([x, y, z])
                    C.append([r, g, b])
    return np.asarray(V, dtype=np.float32), np.asarray(C, dtype=np.float32)


def normalize_vertices(V):
    V = V - V.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(V, axis=1))
    return V / (scale + 1e-8)


def process_obj(obj_path, out_dir):
    # Load XYZ + RGB
    V, C = load_obj_with_vertex_color(obj_path)

    if len(V) == 0:
        raise RuntimeError("No vertices found")

    # Normalize geometry
    V = normalize_vertices(V)

    # Compute normals
    mesh = trimesh.load(obj_path, process=False)
    N = mesh.vertex_normals.astype(np.float32)

    if len(N) != len(V):
        raise RuntimeError("Normals count mismatch")

    points = np.concatenate([V, N], axis=1)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "points.npy"), points)
    np.save(os.path.join(out_dir, "colors.npy"), C)


def process_split(split_name):
    in_root = os.path.join(RAW_ROOT, split_name)
    out_root = os.path.join(OUT_ROOT, split_name)

    obj_files = [
        f for f in os.listdir(in_root)
        if f.lower().endswith(".obj")
    ]

    print(f"\nProcessing {split_name}: {len(obj_files)} OBJ files")

    for obj_file in tqdm(obj_files):
        obj_path = os.path.join(in_root, obj_file)
        case_name = os.path.splitext(obj_file)[0]
        out_dir = os.path.join(out_root, case_name)

        try:
            process_obj(obj_path, out_dir)
        except Exception as e:
            print(f"[ERROR] {obj_file}: {e}")


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    for split in ["upper", "lower"]:
        split_dir = os.path.join(RAW_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split}: folder not found")
            continue
        process_split(split)

    print("\n✅ Dataset preprocessing completed")


if __name__ == "__main__":
    main()
