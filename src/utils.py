# utils.py
#       It includes a custom Dataset class, a function to set the random seed for reproducibility, and a function to compute entropy.

import torch
import numpy as np
import random 
from torch.utils.data import Dataset

def print_framed_metrics(val_loss, val_acc):
    """
    Prints validation loss and accuracy inside a pretty ASCII box.
    """
    msg = f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
    border = '+' + '-' * (len(msg) + 2) + '+'
    print(border)
    print(f"| {msg} |")
    print(border)


def set_seed(seed=42):
    '''
    To have reproducible results EVERY time the code is ran, we set the seed for ALL random number generators.
    This includes numpy, random, and torch.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class ImageDataset(Dataset):
    '''
    A custom dataset class for loading the images and labels from our given dataset.
    Assumes images are in a numpy array format and labels are in a separate numpy array.
    '''
    def __init__(self, images: np.ndarray, labels: np.ndarray=None, transform=None):
        # images: (N, 96, 96, 3)
        # labels: (N, ) or None if unlabeled
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # Normalize

        # Convert from (C, H, W) → (H, W, C) for torchvision
        img = np.transpose(img, (1, 2, 0))

        if self.transform:
            img = self.transform(img)  # e.g., ToPILImage → Flip → ToTensor()

        # Final shape is already CHW because ToTensor() does it
        if self.labels is not None:
            label = self.labels[idx]
            return img, torch.tensor(label, dtype=torch.long)
        else:
            return img



    # def __getitem__(self, idx):
    #     img = self.images[idx].astype(np.float32) # Convert to float32 
    #     # Example: scale images to [0,1], convert to CHW
    #     img = img / 255.0 # Normalize to [0, 1]
    
    #     if img.shape[-1] == 3:  # Assume HWC format
    #         img = np.transpose(img, (2, 0, 1))  # (C, H, W)

    #     # img = np.transpose(img, (2, 0, 1))  # (3,96,96)

    #     if self.transform:
    #         # Apply any custom transforms (e.g. augmentations)
    #         img = self.transform(img)
        
    #     if self.labels is not None:
    #         label = self.labels[idx]
    #         return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.long)
    #     else:
    #         # For unlabeled data
    #         return torch.tensor(img, dtype=torch.float)



def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Computes the entropy H(y_hat) = - sum(p_i log(p_i)) for a 1D array of probabilities.
    """
    eps = 1e-9
    p = np.clip(probabilities, eps, 1.0)
    return -np.sum(p * np.log(p))


