# utils.py
#       It includes a custom Dataset class, a function to set the random seed for reproducibility, and a function to compute entropy.

import torch
import numpy as np
import random 
from torch.utils.data import Dataset
import torch.nn.functional as F

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

def load_problem2_data():
    import numpy as np
    import os

    train_path = os.path.join('data', 'problem2', 'training_data.npz')
    val_path = os.path.join('data', 'problem2', 'evaluation_data.npz')

    training_data = np.load(train_path)
    training_images = training_data['a']
    training_labels = training_data['b']

    val_data = np.load(val_path)
    val_images = val_data['a']
    val_labels = val_data['b']

    print('training_images shape:', training_images.shape)
    print('training_labels shape:', training_labels.shape)
    print('val_images shape:', val_images.shape)
    print('val_labels shape:', val_labels.shape)

    return training_images, training_labels, val_images, val_labels

def compute_npz_mean_std(source):
    """
    Computes per-channel mean and standard deviation from image data.
    
    If source is a string, it is treated as the path to an NPZ file that contains image data under key 'a'.
    If source is a NumPy array, it is assumed to be the image data.
    
    If the images are of type uint8, they are normalized to [0, 1].
    
    Parameters:
        source (str or np.ndarray): File path to an NPZ file or a NumPy ndarray containing image data.
    
    Returns:
        mean (np.array): Per-channel mean.
        std (np.array): Per-channel standard deviation.
    """
    import numpy as np

    if isinstance(source, str):
        data = np.load(source)
        images = data['a']
    elif isinstance(source, np.ndarray):
        images = source
    else:
        raise ValueError("source must be a file path or a numpy ndarray.")

    # Normalize if images are uint8
    if images.dtype == 'uint8':
        images = images.astype(np.float32) / 255.0

    if images.ndim != 4:
        raise ValueError(f"Expected a 4D array but got shape {images.shape}")

    # Determine the image format and compute statistics
    if images.shape[1] == 3:
        # images in (N, C, H, W) format
        mean = np.mean(images, axis=(0, 2, 3))
        std = np.std(images, axis=(0, 2, 3))
    elif images.shape[3] == 3:
        # images in (N, H, W, C) format
        mean = np.mean(images, axis=(0, 1, 2))
        std = np.std(images, axis=(0, 1, 2))
    else:
        raise ValueError(f"Could not determine image format from shape {images.shape}")

    print("Computed mean:", mean)
    print("Computed std:", std)
    return mean, std

class ImageDataset(Dataset):
    """
    Loads images stored in a NumPy array.
    images : (N, 96, 96, 3) or (N, 3, 96, 96)
    labels : (N,) or None
    transform : torchvision transform taking a Tensor(3,96,96)
    device : the torch.device to move tensors to
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray = None, transform=None, device='cpu'):
        self.images = images.astype(np.float32) // 255.0  # Normalize to [0, 1]
        self.labels = labels
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        if img.ndim == 3 and img.shape[-1] == 3:  # HWC → CHW
            img = np.transpose(img, (2, 0, 1))

        img = torch.from_numpy(img)  # float32 tensor in CHW
        if self.transform:
            img = self.transform(img)


        if self.labels is not None:
            return img, torch.tensor(self.labels[idx], dtype=torch.long)
        return img
        # # Move to GPU (or specified device)
        # img = img.to(self.device, non_blocking=True)

        # if self.labels is not None:
        #     label = torch.tensor(self.labels[idx], dtype=torch.long, device=self.device)
        #     return img, label
        # else:
        #     return img



    # def __getitem__(self, idx):
    #     img = self.images[idx].astype(np.float32) / 255.0  # Normalize

    #     # Convert from (C, H, W) → (H, W, C) for torchvision
    #     img = np.transpose(img, (1, 2, 0))

    #     if self.transform:
    #         img = self.transform(img)  # e.g., ToPILImage → Flip → ToTensor()

    #     # Final shape is already CHW because ToTensor() does it
    #     if self.labels is not None:
    #         label = self.labels[idx]
    #         return img, torch.tensor(label, dtype=torch.long)
    #     else:
    #         return img



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

def mc_dropout_inference(model, image, T=10, device='cuda'):
    """
    Perform MC Dropout inference on a single image using T stochastic forward passes.
    Assumes the model is already in .eval() mode with dropout manually enabled.
    
    Args:
        model: PyTorch model with dropout layers.
        image: Input image tensor (C, H, W), unbatched.
        T: Number of stochastic forward passes.

    Returns:
        mean_softmax: Mean predicted probabilities over T passes.
        entropy: Entropy of the mean prediction (measure of uncertainty).
    """
    model.to(device)  # ensure model is on correct device
    image = image.unsqueeze(0).to(device)  # add batch dim: (1, C, H, W)

    softmaxes = []
    with torch.no_grad():
        for _ in range(T):
            output = model(image)
            softmax = F.softmax(output, dim=1).squeeze().cpu()
            softmaxes.append(softmax)

    mean_softmax = torch.stack(softmaxes, dim=0).mean(dim=0)
    entropy = -torch.sum(mean_softmax * torch.log(mean_softmax + 1e-10)).item()
    return mean_softmax, entropy


def enable_dropout(model):
    """ Enable dropout layers during evaluation """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

