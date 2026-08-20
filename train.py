import os
import shutil
import torch
from torch.utils.data import DataLoader

from jaw_dataset import JawProcessedDataset
from model_pointnetpp import PointNetPPColor
from losses import smoothness_loss


# ------------------ config ------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_ROOT = "data/processed/upper"   # для lower поменяй путь
WEIGHTS_DIR = "weights"
N_POINTS = 4096
BATCH_SIZE = 8
EPOCHS = 1000
LR = 1e-3
SMOOTH_LAMBDA = 0.04   # снижено с 0.1 — меньше давит на тонкие цветовые детали (желтизна и т.п.)
K_NEIGHBORS = 16   # k для EdgeConv и smoothness_loss
CHECKPOINT_EVERY = 50   # сохранять веса каждые N эпох, независимо от того, лучшие они или нет
LR_MIN = 1e-5   # минимальный LR к концу обучения (косинусный спад)

# --------------------------------------------


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    dataset = JawProcessedDataset(DATA_ROOT, n_points=N_POINTS)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # ВАЖНО для Windows
        pin_memory=True,
        # drop_last=True: модель использует BatchNorm, который падает
        # на батчах размера 1 (может случиться на последнем неполном батче)
        drop_last=True
    )

    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader is empty. Dataset size={len(dataset)}, "
            f"batch_size={BATCH_SIZE}. Try a smaller BATCH_SIZE or drop_last=False."
        )

    model = PointNetPPColor(k=K_NEIGHBORS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR_MIN
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_l1 = 0.0
        total_smooth = 0.0

        for X, Y in loader:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            pred = model(X)

            loss_l1 = torch.nn.functional.l1_loss(pred, Y)
            loss_smooth = smoothness_loss(X[:, :, :3], pred, k=K_NEIGHBORS)

            loss = loss_l1 + SMOOTH_LAMBDA * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_l1 += loss_l1.item()
            total_smooth += loss_smooth.item()

        scheduler.step()

        avg_loss = total_loss / len(loader)
        avg_l1 = total_l1 / len(loader)
        avg_smooth = total_smooth / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:04d} | Loss: {avg_loss:.6f} "
            f"(l1: {avg_l1:.6f}, smooth: {avg_smooth:.6f}) | LR: {current_lr:.2e}"
        )

        # -------- save BEST (with epoch number in filename) --------
        if avg_loss < best_loss:
            best_loss = avg_loss

            # именной чекпоинт: weights/best_epoch0042_loss0.063400.pth
            best_named_path = os.path.join(
                WEIGHTS_DIR,
                f"best_epoch{epoch:04d}_loss{best_loss:.6f}.pth"
            )
            torch.save(model.state_dict(), best_named_path)

            # плюс всегда актуальный best.pth без номера — удобно для infer_visual.py,
            # чтобы не менять путь к весам после каждого улучшения.
            # Копируем уже сохранённый файл вместо повторного torch.save(),
            # чтобы оба файла были побайтово идентичны.
            best_path = os.path.join(WEIGHTS_DIR, "best.pth")
            shutil.copyfile(best_named_path, best_path)

            print(f"🔥 Best model updated (loss={best_loss:.6f}) -> {best_named_path}")

        # -------- periodic checkpoint (regardless of best) --------
        if CHECKPOINT_EVERY and (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(
                WEIGHTS_DIR,
                f"epoch{epoch:04d}_loss{avg_loss:.6f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved -> {ckpt_path}")

    # -------- save LAST --------
    last_path = os.path.join(WEIGHTS_DIR, "last.pth")
    torch.save(model.state_dict(), last_path)

    print("\n✅ Training finished")
    print(f"Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()