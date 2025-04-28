"""
VGG11 Neural Network Implementation with Batch Normalization

This module implements the VGG11 architecture with batch normalization (VGG11BN),
based on the paper "Very Deep Convolutional Networks for Large-Scale Image Recognition"
by Karen Simonyan and Andrew Zisserman (2014).

VGG11 is characterized by its simplicity: it uses only 3×3 convolutional layers 
stacked on top of each other with increasing depth. Pooling layers are used to 
reduce the spatial dimensions. This implementation adds batch normalization after 
each convolutional layer for improved training stability and performance.

Reference: https://arxiv.org/abs/1409.1556
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG11BN(nn.Module):
    """
    VGG11 architecture with batch normalization.
    
    The VGG11 architecture consists of 8 convolutional layers and 3 fully connected layers.
    This implementation adds batch normalization after each convolutional layer and
    optional dropout in the fully connected layers.
    
    Architecture Overview:
    - 5 blocks of convolutional layers with batch normalization and ReLU activation
    - Max pooling after each block to reduce spatial dimensions
    - 3 fully connected layers (4096 -> 4096 -> num_classes)
    - Optional dropout layers after the first two fully connected layers
    
    Input Size: 96x96 (assumes this based on the flattened dimension calculations)
    Output: Class probabilities
    """
    
    def __init__(self, num_classes=3, dropout=False, in_channels=3):
        """
        Initialize the VGG11BN model.
        
        Args:
            num_classes (int): Number of output classes for classification.
            dropout (bool): Whether to apply dropout in fully connected layers.
                            Helps prevent overfitting when True.
            in_channels (int): Number of input channels (3 for RGB, 1 for grayscale).
        """
        super(VGG11BN, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        
        # ===================== FEATURE EXTRACTION PART =====================
        # This part consists of convolutional layers that extract features from the input image
        # The general pattern is: Conv2d -> BatchNorm2d -> ReLU -> (MaxPool2d)
        
        # Structure explanation:
        # - Each conv layer uses 3x3 kernels with padding=1 to maintain spatial dimensions
        # - Max pooling uses 2x2 kernels with stride=2, reducing dimensions by half
        # - Channels increase progressively: 3->64->128->256->512->512
        
        self.features = nn.Sequential(
            # Block 1: Input(3) -> 64 channels
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),  # Output: 64 x H x W
            nn.BatchNorm2d(64),  # Normalize each channel to stabilize training
            nn.ReLU(inplace=True),  # inplace=True modifies input directly, saving memory
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x H/2 x W/2
            
            # Block 2: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output: 128 x H/2 x W/2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x H/4 x W/4
            
            # Block 3: 128 -> 256 -> 256 channels (two conv layers)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output: 256 x H/4 x W/4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # Output: 256 x H/4 x W/4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x H/8 x W/8
            
            # Block 4: 256 -> 512 -> 512 channels (two conv layers)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # Output: 512 x H/8 x W/8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Output: 512 x H/8 x W/8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 512 x H/16 x W/16
            
            # Block 5: 512 -> 512 -> 512 channels (two conv layers)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Output: 512 x H/16 x W/16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),  # Output: 512 x H/16 x W/16
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 512 x H/32 x W/32
        )
        
        # ===================== CLASSIFIER PART =====================
        # This part consists of fully connected layers that classify the extracted features
        
        # For an input of 96x96, after 5 max pooling layers (each dividing dimensions by 2),
        # the feature map size will be 3x3 (96/32 = 3)
        feature_size = 512 * 3 * 3  # 512 channels x 3 x 3 spatial dimensions
        
        self.classifier = nn.Sequential(
            # First FC layer: flattened features -> 4096
            nn.Linear(feature_size, 4096),
            nn.ReLU(True),
            # Dropout with p=0.5 (50% of neurons are randomly deactivated during training)
            # nn.Identity() is a no-op layer used when dropout=False
            nn.Dropout(p=0.5) if self.dropout else nn.Identity(),
            
            # Second FC layer: 4096 -> 4096
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5) if self.dropout else nn.Identity(),
            
            # Output layer: 4096 -> num_classes
            nn.Linear(4096, self.num_classes)
            # Note: No softmax here as it's typically combined with CrossEntropyLoss
            # which applies softmax internally for numerical stability
        )
        
        # weights are not initialized as per the given task
        # Uncomment the following line if you want to initialize weights
        # self._initialize_weights() 

    def forward(self, x):
        """
        Forward pass of the VGG11BN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Pass input through feature extraction layers
        x = self.features(x)
        
        # Flatten the output for the fully connected layers
        # Changes dimensions from (batch_size, channels, height, width) to (batch_size, channels*height*width)
        x = x.view(x.size(0), -1)
        
        # Pass through classifier layers
        x = self.classifier(x)
        
        return x
    
    def _initialize_weights(self):
        """
        Initialize the weights of the network according to the strategy described in the VGG paper.
        
        For convolutional layers:
            - Weights are drawn from a normal distribution with mean=0, std=0.01
            - Biases are initialized to 0
            
        For fully connected layers:
            - Weights are drawn from a normal distribution with mean=0, std=0.005
            - Biases are initialized to 0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def count_parameters(model):
    """
    Count the total number of trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Create a model instance for testing
    model = VGG11BN(num_classes=3, dropout=True)
    
    # Print model information
    print(f"Model Architecture: VGG11 with Batch Normalization")
    print(f"Dropout enabled: {model.dropout}")
    print(f"Number of classes: {model.num_classes}")
    print(f"Total trainable parameters: {count_parameters(model):,}")
    
    # Test with a sample input
    sample_input = torch.randn(1, 3, 96, 96)  # Batch size 1, 3 channels, 96x96 image
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # You can uncomment the following lines to print the model architecture
    # print("\nModel Architecture:")
    # print(model)