import torch
import torch.nn as nn
import torch.nn.functional as F
from models.SE_Attention import SE
from models.PDPNet import PDPNet

class VeggieNet(nn.Module):
    def __init__(self, num_classes, image_size):
        super(VeggieNet, self).__init__()
                
        self.pw = nn.Conv2d(in_channels=3, out_channels=256, kernel_size=1, padding=0)
                
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)   
        
        self.dw1 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64)
        self.pw1 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        
        self.dw2 = nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=1, groups=128)
        self.pw2 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        
        self.seq = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self.pool,
            self.dw1,
            nn.BatchNorm2d(64),
            self.pw1,
            nn.BatchNorm2d(128),
            nn.ReLU(),
            self.dw2,
            nn.BatchNorm2d(128),
            self.pw2,
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.size = image_size // 8
        
        self.pdp = PDPNet(in_channels=512, image_size=63, num_classes=5)
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x):
        batch1 = self.seq(x)

        batch2 = self.pw(x)
        batch2 = F.adaptive_avg_pool2d(batch2, (batch2.size(2), batch2.size(3)))
        
        x = torch.cat((batch1, batch2), dim=1) # C = 256 + 256
        
        x = self.pdp(x)        
        return x
