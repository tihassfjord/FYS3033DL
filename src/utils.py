"""
Utility functions for deep learning projects.

This module provides helper functions and classes for:
- Data loading and preprocessing
- Model training and evaluation
- Interpretability and visualization
- Reproducibility
"""

import os
import multiprocessing as mp
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from PIL import Image    

# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------

def set_seed(seed=42):
    """
    Set seeds for reproducible results across runs.
    
    This function sets the seed for all random number generators to ensure
    consistent results in PyTorch experiments.
    
    Args:
        seed (int): The seed value to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------------------------------------------------------------
# Dataset and DataLoader Utilities
# ------------------------------------------------------------------------------

class ImageDataset(Dataset):
    """
    Dataset class for loading images stored in a NumPy array.
    
    Handles both (N, H, W, C) and (N, C, H, W) formats, normalizes to [0,1],
    and supports optional transformations.
    
    Args:
        images (np.ndarray): Image array with shape (N, H, W, C) or (N, C, H, W)
        labels (np.ndarray, optional): Label array with shape (N,)
        transform: Optional transforms to apply to images
        device (str): Device to move tensors to ('cpu' or 'cuda')
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray = None, transform=None, device='cpu'):
        self.images = images.astype(np.float32) / 255.0
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
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        else:
            return img


def create_dataloaders(
    train_dataset=None,
    val_dataset=None,
    batch_size=32,
    num_workers=None,
    pin_memory=True,
    mode='both',
    shuffle_train=True,
    shuffle_val=False,
    prefetch_factor=2,
    use_prefetcher=True
):
    """
    Creates optimized PyTorch dataloaders for training and/or validation.

    Args:
        train_dataset (Dataset): Dataset for training
        val_dataset (Dataset): Dataset for validation
        batch_size (int): Batch size for loading
        num_workers (int, optional): CPU workers for data loading. If None, will use CPU count.
        pin_memory (bool): Whether to pin memory (recommended for GPU training)
        mode (str): 'train', 'val', or 'both' – selects which loaders to return
        shuffle_train (bool): Whether to shuffle training data
        shuffle_val (bool): Whether to shuffle validation data
        prefetch_factor (int): Number of batches to prefetch per worker
        use_prefetcher (bool): Whether to use data prefetcher for faster GPU transfers

    Returns:
        dict: Dictionary containing one or both of the keys: 'train', 'val'
    """    
    if num_workers is None:
        # Use CPU count as a reasonable default
        num_workers = min(os.cpu_count(), 8)  # Limit to 8 as more rarely helps
    
    # Common DataLoader parameters
    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'persistent_workers': num_workers > 0,
    }
    
    # Only use prefetch_factor if num_workers > 0
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor

    loaders = {}

    if mode in ['train', 'both']:
        assert train_dataset is not None, "train_dataset must be provided when mode is 'train' or 'both'"
        train_loader = DataLoader(
            train_dataset,
            shuffle=shuffle_train,
            **loader_kwargs
        )
        
        # Wrap with prefetcher if requested and CUDA is available
        if use_prefetcher and torch.cuda.is_available():
            train_loader = DataPrefetcher(train_loader)
            
        loaders['train'] = train_loader

    if mode in ['val', 'both']:
        assert val_dataset is not None, "val_dataset must be provided when mode is 'val' or 'both'"
        val_loader = DataLoader(
            val_dataset,
            shuffle=shuffle_val,
            **loader_kwargs
        )
        
        # Use prefetcher for validation as well if requested
        if use_prefetcher and torch.cuda.is_available():
            val_loader = DataPrefetcher(val_loader)
            
        loaders['val'] = val_loader

    return loaders

class DataPrefetcher:
    """
    Data prefetcher to speed up data loading by prefetching the next batch 
    while the GPU is processing the current batch.
    """
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_data = None
        self.preload()
        
        # Store the original loader length for __len__
        self.loader_len = len(loader) if hasattr(loader, '__len__') else None

    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        except Exception as e:
            # Print the error but allow the training to continue
            print(f"WARNING: Error in DataPrefetcher.preload(): {str(e)}")
            self.next_data = None
            return
        
        # Preload next batch in a non-blocking way
        try:
            with torch.cuda.stream(self.stream):
                if isinstance(self.next_data, list) or isinstance(self.next_data, tuple):
                    self.next_data = [
                        item.cuda(non_blocking=True) if torch.is_tensor(item) else item
                        for item in self.next_data
                    ]
                else:
                    self.next_data = self.next_data.cuda(non_blocking=True)
        except Exception as e:
            print(f"WARNING: Error transferring data to GPU in DataPrefetcher: {str(e)}")
            # Attempt a fallback to synchronous transfer
            try:
                if isinstance(self.next_data, list) or isinstance(self.next_data, tuple):
                    self.next_data = [
                        item.cuda() if torch.is_tensor(item) else item
                        for item in self.next_data
                    ]
                else:
                    self.next_data = self.next_data.cuda()
            except Exception:
                # If even the fallback fails, return None
                self.next_data = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_data is None:
            raise StopIteration
            
        # Use try/except to handle potential errors
        try:
            torch.cuda.current_stream().wait_stream(self.stream)
            data = self.next_data
            self.preload()
            return data
        except Exception as e:
            print(f"ERROR in DataPrefetcher.__next__(): {str(e)}")
            self.preload()  # Try to recover for the next iteration
            raise StopIteration  # Skip this batch
    
    def __len__(self):
        return self.loader_len if self.loader_len is not None else 0
# class DataPrefetcher:
#     """
#     Data prefetcher to speed up data loading by prefetching the next batch 
#     while the GPU is processing the current batch.
    
#     This implementation is inspired by NVIDIA's approach described in:
#     https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    
#     The prefetcher uses CUDA streams to overlap data transfer with computation,
#     which can significantly reduce training time by hiding data loading latency.
    
#     Adapted from PyTorch examples and NVIDIA Apex utility code.
#     """
#     def __init__(self, loader):
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream()
#         self.next_data = None
#         self.preload()

#     def preload(self):
#         try:
#             self.next_data = next(self.loader)
#         except StopIteration:
#             self.next_data = None
#             return
        
#         # Preload next batch in a non-blocking way
#         with torch.cuda.stream(self.stream):
#             if isinstance(self.next_data, list) or isinstance(self.next_data, tuple):
#                 self.next_data = [
#                     item.cuda(non_blocking=True) if torch.is_tensor(item) else item
#                     for item in self.next_data
#                 ]
#             else:
#                 self.next_data = self.next_data.cuda(non_blocking=True)

#     def __iter__(self):
#         return self

#     def __next__(self):
#         torch.cuda.current_stream().wait_stream(self.stream)
#         data = self.next_data
#         if data is None:
#             raise StopIteration
#         self.preload()
#         return data
    
#     def __len__(self):
#         return len(self.loader)
# ------------------------------------------------------------------------------
# Data Loading and Processing
# ------------------------------------------------------------------------------

def load_problem2_data():
    """
    Load the data for problem 2 from NPZ files.
    
    Returns:
        tuple: Training images, training labels, validation images, validation labels
    """
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
    
    Args:
        source (str or np.ndarray): File path to NPZ file or NumPy array with image data
    
    Returns:
        tuple: (mean, std) arrays with per-channel statistics
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

    return mean, std


def load_images_as_numpy(root_dir, image_size=(224, 224)):
    """
    Loads all images in a directory (recursively), resizes, and converts to NumPy array.
    
    Args:
        root_dir (str): Root directory to search for images
        image_size (tuple): Target size (width, height) for images
        
    Returns:
        np.ndarray: Array of images with shape (N, H, W, C)
    """
    image_paths = glob(os.path.join(root_dir, '**', '*.*'), recursive=True)
    images = []

    for path in image_paths:
        try:
            img = Image.open(path).convert('RGB').resize(image_size)
            img_np = np.array(img).astype(np.float32) / 255.0  # (H, W, C), scaled
            images.append(img_np)
        except Exception as e:
            print(f"Skipped {path}: {e}")

    # Result: shape (N, H, W, C)
    return np.stack(images)


# ------------------------------------------------------------------------------
# Uncertainty Estimation and Interpretation
# ------------------------------------------------------------------------------

def compute_entropy(probabilities: np.ndarray) -> float:
    """
    Computes the entropy H(y_hat) = - sum(p_i log(p_i)) for a 1D array of probabilities.
    
    Args:
        probabilities (np.ndarray): Array of probability values
        
    Returns:
        float: Entropy value
    """
    eps = 1e-9
    p = np.clip(probabilities, eps, 1.0)
    return -np.sum(p * np.log(p))


def mc_dropout_inference(model, image, T=10, device='cuda'):
    """
    Perform MC Dropout inference on a single image using T stochastic forward passes.
    
    This function assumes the model is already in .eval() mode with dropout manually enabled.
    
    Args:
        model: PyTorch model with dropout layers
        image: Input image tensor (C, H, W), unbatched
        T (int): Number of stochastic forward passes
        device (str): Device to run model on

    Returns:
        tuple: (mean_softmax, entropy) where mean_softmax is the mean predicted 
               probabilities over T passes, and entropy is a measure of uncertainty
    """
    model.to(device)  
    model.eval()  # Ensure model is in eval mode
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
    """
    Enable dropout layers during evaluation for uncertainty estimation.
    
    Args:
        model: PyTorch model with dropout layers
    """
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def visualize_occlusion_analysis(model, image_tensor, label, class_names, occlusion_size=10, baseline_tf=None, device='cuda'):
    """
    Performs occlusion analysis on an image and visualizes the results.
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (CxHxW)
        label: True label index
        class_names: List of class names
        occlusion_size: Size of the occlusion patch
        baseline_tf: Transform to apply to occluded images
        device: Device to run model on ('cuda' or 'cpu')
    
    Returns:
        tuple: (heatmap, class_pixels, counters) showing confidence drop, class predictions,
               and class prediction counts during occlusion
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import BoundaryNorm
    from matplotlib.colorbar import ColorbarBase
    from matplotlib import cm
    from collections import defaultdict
    import matplotlib.pyplot as plt
    
    # Input tensor for model prediction
    input_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Original image data for display
    img = image_tensor.permute(1, 2, 0).cpu().numpy()  # Move to CPU and permute to HWC format for imshow
    
    # Display the original image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Original Image: {class_names[label]}")
    plt.axis('off')
    plt.show()
    
    # Get original prediction without occlusion
    model.eval()
    with torch.no_grad():
        original_pred = model(input_tensor).cpu().numpy()
        original_prob = np.exp(original_pred) / np.sum(np.exp(original_pred))
        pred_class = np.argmax(original_pred[0])
        
    print(f"Original prediction: {class_names[pred_class]} (confidence: {original_prob[0][pred_class]:.4f})")
    
    # Image dimensions
    img_size = img.shape[0]  # Size of the image
    
    print('Running occlusion analysis...')
    
    # Create heatmap to store importance of each region
    heatmap = np.zeros((img_size, img_size), np.float32)
    
    # Store the predicted class for each occluded region
    class_pixels = np.zeros((img_size, img_size), np.int16)
    
    # Keep track of prediction counts
    counters = defaultdict(int)
    
    # Run occlusion analysis
    for n, (x, y, occluded_img) in enumerate(iter_occlusion(img, size=occlusion_size)):
        if n % 50 == 0:
            print(f'Processing occlusion {n}...')
            
        # Convert the occluded image to a tensor and normalize it
        occluded_tensor = torch.from_numpy(occluded_img).permute(2, 0, 1).float()
        if baseline_tf:
            occluded_tensor = baseline_tf(occluded_tensor)
        occluded_tensor = occluded_tensor.unsqueeze(0).to(device)
        
        # Get model prediction for occluded image
        with torch.no_grad():
            occluded_pred = model(occluded_tensor).cpu().numpy()
            
        # Store the change in confidence for the correct class
        confidence_drop = original_pred[0][label] - occluded_pred[0][label]
        heatmap[y:y + occlusion_size, x:x + occlusion_size] = confidence_drop
        
        # Store the predicted class for this occluded region
        pred_class = np.argmax(occluded_pred[0])
        class_pixels[y:y + occlusion_size, x:x + occlusion_size] = pred_class
        counters[pred_class] += 1
    
    # Normalize the heatmap for better visualization
    heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Plot the results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_normalized, cmap='hot')
    plt.title("Occlusion Sensitivity Map")
    plt.colorbar(label='Normalized Confidence Drop')
    plt.axis('off')
    
    # Overlay heatmap on original image
    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(heatmap_normalized, cmap='hot', alpha=0.5)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create detailed visualization
    fig = plt.figure(figsize=(12, 5))
    
    # First subplot - heatmap showing confidence drop
    ax1 = plt.subplot(1, 2, 1, aspect='equal')
    hm = ax1.imshow(heatmap)
    ax1.set_title("Confidence Drop when Region is Occluded")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Second subplot - class prediction when region is occluded
    ax2 = plt.subplot(1, 2, 2, aspect='equal')
    vals = np.unique(class_pixels).tolist()
    bounds = vals + [vals[-1] + 1]  # add an extra item for the colormap boundaries
    
    # Create a custom colormap with distinct colors for each class
    custom = cm.get_cmap('tab10', len(bounds)) # using tab10 for clearer distinction
    norm = BoundaryNorm(bounds, custom.N)
    
    cp = ax2.imshow(class_pixels, norm=norm, cmap=custom)
    ax2.set_title("Class Prediction when Region is Occluded")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Add colorbar for the heatmap
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(hm, cax=cax1)
    cbar1.set_label('Confidence Drop')
    
    # Add colorbar for the class predictions
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar2 = ColorbarBase(cax2, cmap=custom, norm=norm,
                        ticks=[(a + b) / 2.0 for a, b in zip(bounds[:-1], bounds[1:])],
                        boundaries=bounds, spacing='uniform', orientation='vertical')
    
    # Set the labels for each class in the colorbar
    class_labels = [class_names[i] if i < len(class_names) else f"Class {i}" for i in vals]
    cbar2.ax.set_yticklabels(class_labels)
    cbar2.set_label('Predicted Class')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary of confidence drop and class predictions
    print(f"Original image class: {class_names[label]}")
    print("\nMost predicted classes during occlusion:")
    for class_id, count in sorted(counters.items(), key=lambda x: -x[1]):
        percentage = (count / sum(counters.values())) * 100
        print(f"{class_names[class_id]}: {count} occurrences ({percentage:.2f}%)")
        
    return heatmap, class_pixels, counters


# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------

def print_framed_metrics(val_loss, val_acc):
    """
    Prints validation loss and accuracy inside a pretty ASCII box.
    
    Args:
        val_loss (float): Validation loss
        val_acc (float): Validation accuracy
    """
    msg = f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}"
    border = '+' + '-' * (len(msg) + 2) + '+'
    print(border)
    print(f"| {msg} |")
    print(border)


def iter_occlusion(img, size=10, stride=None):
    """
    Generate occluded versions of an image for occlusion sensitivity analysis.
    
    Args:
        img: Input image (H, W, C)
        size (int): Size of occlusion patch
        stride (int, optional): Stride for occlusion window
        
    Yields:
        tuple: (x, y, occluded_image) containing the position and resulting occluded image
    """
    if stride is None:
        stride = size // 2
        
    h, w = img.shape[:2]
    
    # Create a copy of the image for occlusion
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            occluded = img.copy()
            # Create a gray occlusion patch (could use black, mean color, etc.)
            occluded[y:y + size, x:x + size, :] = 0.5
            yield x, y, occluded


if __name__ == "__main__":
    print("This is a utility module. It is not meant to be run directly.")