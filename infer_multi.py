import os
import re
import numpy as np
import torch
import open3d as o3d

from model_pointnetpp import PointNetPPColor
from geometry_utils import normalize_vertices
from utils.segmented_obj_utils import (
    load_segmented_obj, compute_vertex_normals, fix_mesh_orientation,
    save_colored_obj, save_colored_obj_grouped
)


# Загружает обученные веса и красит скан - сегментированный (с группами
# o tooth_N / o gingiva) ИЛИ обычный единый меш без сегментации, автоматически.
# Если группы есть - точная маска зуб/десна берётся прямо из структуры файла;
# если групп нет - модель работает по одной геометрии (как и обучалась для
# этого случая через label_dropout в train.py). Сохраняет результат и
# опционально открывает 3D-визуализацию.
#
# n_passes: сколько раз прогнать весь скан со случайной перестановкой точек
# перед разбиением на чанки по n_points, с усреднением предсказаний по
# каждой точке. Больше проходов = точнее (меньше шанс, что точка окажется
# в "неудачном" чанке), но дольше. 4 - разумный баланс по умолчанию.

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


def extract_epoch_suffix(model_path):
    """
    Достаёт номер эпохи из имени файла весов, например:
        best_epoch0042_loss0.063400.pth -> "0042"
        epoch0049_loss0.065334.pth      -> "0049"
        best.pth / last.pth             -> None (номера эпохи в имени нет)
    """
    name = os.path.basename(model_path)
    match = re.search(r"epoch(\d+)", name)
    return match.group(1) if match else None


def add_epoch_suffix(out_path, model_path):
    """
    Вставляет номер эпохи перед расширением файла:
        result_colored_upper.obj + best_epoch0000_loss...pth
        -> result_colored_upper_0000.obj
    Если номер эпохи извлечь не удалось (например, best.pth/last.pth без
    номера), возвращает out_path без изменений.
    """
    epoch_suffix = extract_epoch_suffix(model_path)
    if epoch_suffix is None:
        return out_path

    root, ext = os.path.splitext(out_path)
    return f"{root}_{epoch_suffix}{ext}"


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
    n_passes=4
):
    # ---------- load mesh: сегментированный (o tooth_N/gingiva) или обычный ----------
    V, F, _C_unused, label, tooth_id = load_segmented_obj(obj_path)

    if len(V) == 0:
        raise ValueError("OBJ has no vertices")
    if len(F) == 0:
        raise ValueError("OBJ has no faces (needed to compute normals and export)")

    is_segmented = (label >= 0).any()
    print(f"[i] Вход {'сегментирован' if is_segmented else 'не сегментирован'} "
          f"({(label >= 0).sum()}/{len(label)} вершин с меткой)")

    # ---------- normalize (та же функция, что и в prepare_dataset.py) ----------
    Vn, center, scale = normalize_vertices(V)

    # ---------- normals (согласуем winding order по всему мешу перед расчётом) ----------
    # Та же коррекция, что и в prepare_dataset.py - иначе часть нормалей
    # (и, соответственно, вход сети) окажется развёрнута на 180°.
    F = fix_mesh_orientation(V, F)
    N = compute_vertex_normals(V, F)

    # ---------- mask + label ----------
    # mask=1 там, где вершина реально входит в какую-то группу (o tooth_N
    # или o gingiva), mask=0 там, где сегментации нет (несегментированный
    # файл целиком, либо отдельные "бесхозные" вершины на швах).
    mask = (label >= 0).astype(np.float32)
    label_clean = np.where(label >= 0, label, 0).astype(np.float32)

    # ---------- model input: xyz + normals + segment_label + mask = 8 признаков ----------
    X = np.concatenate(
        [Vn, N, label_clean.reshape(-1, 1), mask.reshape(-1, 1)], axis=1
    )  # (N, 8)

    # ---------- load model ----------
    model = PointNetPPColor(k=k_neighbors).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # ---------- inference (несколько случайных проходов + усреднение) ----------
    # ВАЖНО: при обучении (jaw_dataset.py) сеть видит n_points случайно
    # выбранных точек ПО ВСЕМУ скану (np.random.choice), а не куски по
    # порядку вершин в файле. Если резать вход последовательными кусками
    # X[i:i+bs] (как раньше), часть чанков окажется однородной (например,
    # чанк только из десны или только из одного зуба целиком) - сеть
    # никогда не видела такого на обучении и даёт "плоское" предсказание
    # именно на таких чанках. Отсюда "полосатая" раскраска (часть зубов
    # без цвета).
    #
    # Чтобы честно повторить распределение обучения, делаем несколько
    # проходов со случайной перестановкой точек, разбитых на чанки по
    # n_points, и усредняем предсказания по каждой точке.
    n = len(X)
    preds_sum = np.zeros((n, 3), dtype=np.float32)
    preds_count = np.zeros((n, 1), dtype=np.float32)

    for pass_i in range(n_passes):
        perm = np.random.permutation(n)

        for i in range(0, n, n_points):
            idx = perm[i:i + n_points]
            chunk = X[idx]

            if len(chunk) < n_points:
                pad = n_points - len(chunk)
                chunk = np.pad(chunk, ((0, pad), (0, 0)), mode="edge")

            chunk_t = torch.from_numpy(chunk).unsqueeze(0).to(device)  # (1, n_points, 8)
            pred = model(chunk_t)[0].cpu().numpy()                     # (n_points, 3)

            valid_len = len(idx)
            preds_sum[idx] += pred[:valid_len]
            preds_count[idx] += 1.0

    preds = preds_sum / np.maximum(preds_count, 1.0)
    preds = np.clip(preds, 0.0, 1.0)

    # ---------- assign colors & export (в исходных, не нормализованных координатах) ----------
    out_path = add_epoch_suffix(out_path, model_path)
    save_colored_obj_grouped(out_path, V, F, preds, label, tooth_id)

    print(f"✅ Saved colored mesh: {out_path}")

    # ---------- visualize ----------
    if visualize:
        visualize_open3d(V, preds)


# ---------- entry point ----------

if __name__ == "__main__":
    model_path = "weights_l/best_epoch0073_loss0.064479.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Пары: (входной_неокрашенный, выходной_окрашенный)
    jobs = [
        ("test/result_uncolored_lower.obj",   "result/result_colored_lower.obj"),
        ("test/result_seg_uncolored_lower.obj", "result/result_seg_colored_lower.obj"),
    ]

    for obj_path, out_path in jobs:
        print(f"\n{'='*50}")
        print(f"▶ Processing: {obj_path}")
        print(f"{'='*50}")

        infer(
            obj_path=obj_path,
            model_path=model_path,
            out_path=out_path,
            n_points=4096,
            k_neighbors=16,
            device=device,
            visualize=True,      # ← False, если не хотите, чтобы окна визуализации
            n_passes=4           #     открывались между файлами (Open3D блокирует)
        )