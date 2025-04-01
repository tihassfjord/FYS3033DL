# %% [utils.py]
# NOTE: This cell simulates the contents of a file named utils.py
#       It includes a custom Dataset class, a stratified split function, etc.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from typing import Tuple, List

class ImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray=None, transform=None):
        # images: (N, 96, 96, 3)
        # labels: (N, ) or None if unlabeled
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32)
        # Example: scale images to [0,1], convert to CHW
        img = img / 255.0
        # img = np.transpose(img, (2, 0, 1))  # (3,96,96)

        if self.transform:
            # Apply any custom transforms (e.g. augmentations)
            img = self.transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        else:
            # For unlabeled data
            return torch.tensor(img, dtype=torch.float)

def split_dataset_stratified(images: np.ndarray, labels: np.ndarray,
                            val_fraction: float=0.2) -> Tuple[List[int], List[int]]:
    """
    Splits indices into train/val sets with class stratification.
    """
    # Basic example of stratified split:
    num_classes = len(np.unique(labels))
    train_indices = []
    val_indices = []

    for c in range(num_classes):
        c_indices = np.where(labels == c)[0]
        np.random.shuffle(c_indices)
        val_size = int(len(c_indices) * val_fraction)
        val_indices.extend(c_indices[:val_size])
        train_indices.extend(c_indices[val_size:])

    return train_indices, val_indices

def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Computes the entropy H(y_hat) = - sum(p_i log(p_i)) for a 1D array of probabilities.
    """
    eps = 1e-9
    p = np.clip(probabilities, eps, 1.0)
    return -np.sum(p * np.log(p))

def plot_entropy_hist(entropies_known, entropies_unknown, bins=30):
    """
    Plots two histograms on the same figure.
    """
    plt.figure()
    plt.hist(entropies_known, bins=bins, alpha=0.5, label='Known Class Entropy')
    plt.hist(entropies_unknown, bins=bins, alpha=0.5, label='Unknown Class Entropy')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.title('Entropy Histograms for Known vs. Unknown')
    plt.legend()
    plt.show()
