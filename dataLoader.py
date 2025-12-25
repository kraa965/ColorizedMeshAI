from jaw_dataset import JawProcessedDataset
from torch.utils.data.dataloader import DataLoader

train_ds = JawProcessedDataset(
    root_dir="data/processed/upper",
    n_points=4096
)

train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

for X, Y in train_loader:
    print(X.shape, Y.shape)
    break
