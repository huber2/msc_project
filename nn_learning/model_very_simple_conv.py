import torch
import torch.nn as nn

class ConvNet(nn.Module):   
    def __init__(self, n_classes):
        super().__init__()

        self.features = nn.Sequential(
            # 16x16 --> 16x16 --> 8x8
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 8x8 --> 8x8 --> 4x4
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # 4x4 --> 4x4 --> 2x2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # 2x2x64 --concat--> 256 --> 16 --> n_classes (3 or 6)
        self.fc = nn.Sequential(
            nn.Linear(256, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, n_classes),
        )  

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x