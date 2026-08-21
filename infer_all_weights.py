import os
import glob
import torch

from infer_visual import infer


# прогоняет один и тот же тестовый скан (сегментированный или обычный)
# через КАЖДЫЙ файл весов из папки weights и сохраняет каждый результат
# отдельным .obj в out_dir. Визуализация не открывается.

def infer_all_weights(
    obj_path,
    weights_dir,
    out_dir,
    n_points=4096,
    k_neighbors=16,
    device="cuda",
    pattern="*.pth"
):
    """
    Прогоняет один и тот же obj_path через каждый файл весов из weights_dir
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
            visualize=False   # визуализация отключена при массовом прогоне
        )

    print(f"\n✅ Готово: {len(weight_paths)} результатов сохранено в {out_dir}")


# ---------- entry point ----------

if __name__ == "__main__":
    obj_path = "test/upper1.obj"  # сегментированный или обычный - определится автоматически
    weights_dir = "weights"
    out_dir = "test/results_by_weights"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    infer_all_weights(
        obj_path=obj_path,
        weights_dir=weights_dir,
        out_dir=out_dir,
        n_points=4096,
        k_neighbors=16,
        device=device,
        pattern="*.pth"   # можно сузить, например "best_epoch*.pth", чтобы пропустить best.pth/last.pth
    )
