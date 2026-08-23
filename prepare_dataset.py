import os
import numpy as np
from tqdm import tqdm

# готовит датасет: читает сегментированные .obj со сканами (группы o gingiva /
# o tooth_N, с цветом в v-строках), вытаскивает точную маску зуб/десна из
# структуры файла, считает нормали, нормализует геометрию, сохраняет
# points.npy/colors.npy в data/processed/...

from geometry_utils import normalize_vertices
from utils.segmented_obj_utils import load_segmented_obj, compute_vertex_normals, fix_mesh_orientation


RAW_ROOT = "data/raw"          # только сегментированные .obj (с группами o/g)
OUT_ROOT = "data/processed"


def process_obj(obj_path, out_dir):
    V, F, C, label, tooth_id = load_segmented_obj(obj_path)

    if len(V) == 0:
        raise RuntimeError("No vertices found")
    if C is None:
        raise RuntimeError("OBJ has no embedded vertex colors (v x y z r g b)")
    if len(F) == 0:
        raise RuntimeError("No faces found (needed to compute normals)")

    # ---------- fix color range ----------
    # Некоторые экспортёры пишут цвет 0-255 вместо 0-1 по спеке OBJ.
    if C.max() > 1.0 + 1e-6:
        C = C / 255.0
    C = np.clip(C, 0.0, 1.0)

    # ---------- убрать неразмеченные вершины (label == -1) ----------
    # Такие вершины не входят ни в одну грань ни одной группы -
    # исключаем их из обучающих данных, чтобы не путать сеть.
    valid = label >= 0
    if not valid.all():
        dropped = (~valid).sum()
        print(f"  [!] {os.path.basename(obj_path)}: dropping {dropped} unlabeled vertices")

    # ---------- normalize geometry ----------
    Vn, _, _ = normalize_vertices(V)

    # ---------- normals (согласуем winding order по всему мешу перед расчётом) ----------
    # Без этого шага нормали для части граней могут получиться развёрнутыми
    # на 180° из-за несогласованного порядка вершин в исходном OBJ (см.
    # диагностику: tooth_21-28 были развёрнуты на 100% ровно по одной
    # половине зубного ряда) - неверный знак нормали портит признак,
    # на котором учится сеть.
    F = fix_mesh_orientation(V, F)
    N = compute_vertex_normals(V, F)

    # points: xyz + normals + label (зуб=1/десна=0) + mask (1=метка известна)
    # = 8 признаков на точку. При подготовке датасета метка всегда реальная
    # (mask=1) - "выключение" метки для эмуляции несегментированного входа
    # делается позже, во время обучения, в jaw_dataset.py (аугментация).
    mask = np.ones((len(V), 1), dtype=np.float32)
    points = np.concatenate(
        [Vn, N, label.reshape(-1, 1).astype(np.float32), mask], axis=1
    )

    points = points[valid]
    C = C[valid]

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "points.npy"), points.astype(np.float32))  # (N, 8)
    np.save(os.path.join(out_dir, "colors.npy"), C.astype(np.float32))       # (N, 3)


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
    print("   points.npy теперь (N, 8): xyz(3) + normals(3) + label(1, зуб=1/десна=0) + mask(1)")


if __name__ == "__main__":
    main()