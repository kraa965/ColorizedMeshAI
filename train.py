import os
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
SMOOTH_LAMBDA = 0.1

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
        drop_last=False
    )

    if len(loader) == 0:
        raise RuntimeError(
            f"DataLoader is empty. Dataset size={len(dataset)}, "
            f"batch_size={BATCH_SIZE}"
        )

    model = PointNetPPColor().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for X, Y in loader:
            X = X.to(DEVICE)
            Y = Y.to(DEVICE)

            pred = model(X)

            loss_l1 = torch.nn.functional.l1_loss(pred, Y)
            loss_smooth = smoothness_loss(X[:, :, :3], pred)

            loss = loss_l1 + SMOOTH_LAMBDA * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch:04d} | Loss: {avg_loss:.6f}")

        # -------- save BEST only --------
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(WEIGHTS_DIR, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"🔥 Best model updated (loss={best_loss:.6f})")

    # -------- save LAST --------
    last_path = os.path.join(WEIGHTS_DIR, "last.pth")
    torch.save(model.state_dict(), last_path)

    print("\n✅ Training finished")
    print(f"Best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
