import gzip
import os
import typing

import numpy as np
import torch


def load_mnist(path, kind='train') -> typing.Tuple[np.ndarray, np.ndarray]:
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 1, 28, 28)

    return images, labels

# it is important our custom class subclasses the Dataset class
class CustomFashinMNSITDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, kind: str = 'train') -> None:
        self.X, self.y = load_mnist(path, kind)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], torch.zeros(10, dtype=torch.float).scatter_(0, self.y[index], value=1)
