# utils.py: Provides utility functions including a custom Dataset class.
# This Dataset class is designed for Problem 2: it handles image normalization and formatting.

import numpy as np
import torch
from torch.utils.data import Dataset

class Problem2Dataset(Dataset):
    """
    A custom Dataset for loading images and labels.
    
    The images are normalized to the range [0, 1] and converted from HWC (Height, Width, Channel)
    format (common in NumPy arrays) to CHW (Channel, Height, Width) format required by PyTorch.
    """
    def __init__(self, images, labels, transform=None):
        # images: NumPy array of shape (N, H, W, C)
        # labels: List or array of labels for each image.
        # transform: Optional transform to apply to each image.
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # Returns the total number of samples.
        return len(self.labels)

    def __getitem__(self, idx):
        # Retrieves the image and label at the given index.
        # Normalize image pixel values from 0-255 to 0-1.
        image = self.images[idx].astype(np.float32) / 255.0
        # Convert image from HWC to CHW format.
        image = image.transpose(2, 0, 1)
        # Convert the NumPy array to a PyTorch tensor.
        image = torch.from_numpy(image)
        # Get the label and convert to integer.
        label = int(self.labels[idx])
        
        # Apply any additional transformation if provided.
        if self.transform:
            image = self.transform(image)
        
        return image, label

if __name__ == "__main__":
    # Quick test of the custom dataset.
    # Create dummy data: 100 images of 32x32 pixels with 3 channels (RGB).
    images = np.random.rand(100, 32, 32, 3)
    # Create dummy labels: Random integers from 0 to 9 (10 classes).
    labels = np.random.randint(0, 10, size=(100,))
    
    dataset = Problem2Dataset(images, labels)
    print(f"Dataset length: {len(dataset)}")
    
    # Retrieve and display details of the first image.
    first_image, first_label = dataset[0]
    print(f"First image shape: {first_image.shape} (Expected: [3, 32, 32]), label: {first_label}")
    # Check if the image is normalized between 0 and 1.
    print(f"First image min: {first_image.min()}, max: {first_image.max()} (Expected: [0, 1])")