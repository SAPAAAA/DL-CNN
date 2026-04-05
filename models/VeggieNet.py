import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SE_Attention import SE
from models.PDPNet import PDPNet

class VeggieNet(nn.Module):
    def __init__(self, num_classes, image_size):
        super(VeggieNet, self).__init__()
        
        size = (image_size - 2) // 4
                
        self.seq = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=2, padding=0),
            nn.Conv2d(128, 64, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * size ** 2, num_classes)
        )        

    def forward(self, x):
        x = self.seq(x)        
        return x
