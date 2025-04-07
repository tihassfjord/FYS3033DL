# %% [model.py]
# NOTE: This cell simulates the contents of a file named model.py
#       You can copy-paste it into an actual file if desired.

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG11BN(nn.Module):
    def __init__(self, num_classes=3, dropout=False):
        super(VGG11BN, self).__init__()
        self.num_classes = num_classes
        self.dropout = dropout

        # Convolutional part (features)
        # Configuration: 64 -> MP -> 128 -> MP -> 256 x2 -> MP -> 512 x2 -> MP -> 512 x2 -> MP
        # After each conv, we insert a batchnorm.
        # Pool kernel size is 2, stride 2.

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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
        # Typically for VGG: 4096 -> 4096 -> num_classes
        # applies dropout(0.5) if self.dropout==True.

        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(True),
            # Dropout to be optionally inserted:
            nn.Dropout(p=0.5) if self.dropout else nn.Identity(),

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
    # Create a model instance for 3 classes (planes, ships, trucks) as required.
    model_2a = VGG11BN(num_classes=3, dropout=False)
    print(model_2a)

