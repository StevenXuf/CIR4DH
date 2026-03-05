from typing import Callable, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# 1) Custom Dataset example
# -------------------------
class MyDataset(Dataset):
    """
    Dataset contract:
      - __len__ returns number of samples
      - __getitem__ returns one sample (and label) at index

    You can pass in transforms for x/y separately.
    """
    def __init__(
        self,
        X: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        x_transform: Optional[Callable] = None,
        y_transform: Optional[Callable] = None,
    ):
        if not torch.is_tensor(X):
            raise TypeError("X must be a torch.Tensor")
        if y is not None and (not torch.is_tensor(y) or len(y) != len(X)):
            raise ValueError("y must be a torch.Tensor with the same length as X")

        self.X = X
        self.y = y
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.x_transform is not None:
            x = self.x_transform(x)

        if self.y is None:
            return x

        y = self.y[idx]
        if self.y_transform is not None:
            y = self.y_transform(y)

        return x, y


def get_loaders(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    build DataLoaders.
    """
    n = len(dataset)

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )

    return val_loader
