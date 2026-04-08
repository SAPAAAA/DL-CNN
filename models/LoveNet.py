import torch
import torch.nn as nn

class LoveNet(nn.Module):
    def __init__(self, image_size, num_classes=5):
        super(LoveNet, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, groups=64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, groups=128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        size = image_size // 8
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * size ** 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.seq(x)
        x = self.head(x)
        return x
