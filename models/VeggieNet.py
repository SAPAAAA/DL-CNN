import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SE_Attention import SE
from models.PDPNet import PDPNet

class VeggieNet(nn.Module):
    def __init__(self, num_classes, image_size):
        super(VeggieNet, self).__init__()
                
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2, groups=128),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.seq2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((28, 28))
        )
        
        self.pdp = PDPNet(in_channels=512, image_size=28, num_classes=num_classes)
        

    def forward(self, x):
        batch1 = self.seq1(x)
        batch2 = self.seq2(x)
        
        x = torch.cat((batch1, batch2), dim=1)
        
        x = self.pdp(x)        
        return x
