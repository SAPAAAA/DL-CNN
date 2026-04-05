import torch
import torch.nn as nn
import torch.nn.functional as F

class VeggieNet(nn.Module):
    def __init__(self, num_classes, image_size):
        super(VeggieNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dw1 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16)
        self.pw1 = nn.Conv2d(16, 32, kernel_size=1, padding=0)
        
        self.dw2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
        self.pw2 = nn.Conv2d(32, 64, kernel_size=1, padding=0)
        
        self.size = image_size // 8
        
        self.fc1 = nn.Linear(64 * 28 * 28, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dw1(x)
        x = F.relu(self.pw1(x))
        x = self.dw2(x)
        x = F.relu(self.pw2(x))
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
