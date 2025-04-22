import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG11BN(nn.Module):
    def __init__(self, num_classes=3, dropout=False, in_channels=3):
        """
        VGG11 with batch normalization and dropout.
        Args:
            num_classes (int): Number of output classes.
            dropout (bool): Whether to apply dropout or not.
            in_channels (int): Number of input channels (default is 3 for RGB images).
        """
        super(VGG11BN, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        # Convolutional part (features)
        # The padding is 1 to keep the spatial dimensions the same after convolution.
        # The stride is 1 by default.
        # Configuration: 64 -> MP -> 128 -> MP -> 256 x2 -> MP -> 512 x2 -> MP -> 512 x2 -> MP
        # After each conv, we insert a batchnorm.
        # Pool kernel size is 2, stride 2.

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier part
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),          # (512,1,1)
        #     nn.Flatten(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(0.5) if self.dropout else nn.Identity(),
        #     nn.Linear(512, num_classes)
        # )

        # Typically for VGG: 4096 -> 4096 -> num_classes
        # applies dropout(0.5) if self.dropout==True.

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(True),
            # Dropout to be optionally inserted:
            nn.Dropout(p=0.5) if self.dropout else nn.Identity(),  # nn.identity() is a "pass-through" layer if dropout is False.
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            # Dropout to be optionally inserted:
            nn.Dropout(p=0.5) if self.dropout else nn.Identity(),  

            nn.Linear(4096, self.num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Create a model instance for 3 classes with dropout
    model = VGG11BN(num_classes=3, dropout=True)
    print(f'INFO : Model created with dropout={model.dropout}, Total parameters: {sum(p.numel() for p in model.parameters())}')

    



