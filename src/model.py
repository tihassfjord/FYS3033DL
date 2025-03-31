# model.py: Defines a VGG-11 model with Batch Normalization and Dropout layers.
# This network is used for image classification on three classes (plane, ship, truck).

import torch.nn as nn

class VGG11_BN_Dropout(nn.Module):
    def __init__(self, num_classes=3):
        super(VGG11_BN_Dropout, self).__init__()
        # 'features' holds the convolutional layers.
        # We follow the VGG-11 structure and add BatchNorm for stability.
        self.features = nn.Sequential(
            # Block 1: from 3 (RGB) to 64 channels.
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Convolution: extracts features.
            nn.BatchNorm2d(64),                          # Normalizes outputs to speed up training.
            nn.ReLU(inplace=True),                       # Applies non-linearity.
            nn.MaxPool2d(2),                             # Reduces spatial size by 2.

            # Block 2: from 64 to 128 channels.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: from 128 to 256 channels.
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 4: maintains 256 channels, then pool.
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: from 256 to 512 channels.
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 6: maintains 512 channels, then pool.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 7: stays at 512 channels.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Block 8: final block with pooling.
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 'classifier' holds the fully connected layers.
        # Dropout layers are added to reduce overfitting by randomly disabling neurons.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),  # Flattened features to 4096 neurons.
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),             # Dropout layer to randomly drop 50% of neurons.
            
            nn.Linear(4096, 4096),         # Second fully connected layer.
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),             # Another Dropout layer.
            
            nn.Linear(4096, num_classes)   # Final layer outputs the class scores.
        )

    def forward(self, x):
        # Forward pass: how input data flows through the network.
        x = self.features(x)            # Pass input through convolutional layers.
        x = x.view(x.size(0), -1)         # Flatten the 2D feature maps into 1D vector.
        x = self.classifier(x)            # Pass through the fully connected classifier.
        return x

if __name__ == "__main__":
    # If you run this file directly, print the network architecture.
    model = VGG11_BN_Dropout()
    print(model)
