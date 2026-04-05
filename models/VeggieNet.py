import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PDPNet import PDPNet

class VeggieNet(nn.Module):
    def __init__(self, num_classes, image_size):
        super(VeggieNet, self).__init__()
                
        self.pdp = PDPNet(in_channels=3, image_size=image_size, num_classes=3)
        self.pw = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, padding=0)
                
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)   
        
        self.dw1 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, groups=16)
        self.pw1 = nn.Conv2d(16, 32, kernel_size=1, padding=0)
        
        self.dw2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, groups=32)
        self.pw2 = nn.Conv2d(32, 64, kernel_size=1, padding=0)
        
        self.seq = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            self.pool,
            self.dw1,
            self.pw1,
            nn.ReLU(),
            self.dw2,
            self.pw2,
            nn.ReLU()
        )
        
        self.size = image_size // 8
        
        self.fc1 = nn.Linear(128 * self.size ** 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        

    def forward(self, x):
        
        pdp = self.pdp(x)
        scale = pdp.view(pdp.size(0), pdp.size(1), 1, 1)
        x_pdp = x * scale
        
        batch2 = self.seq(x)
        x_pdp = x + x_pdp
        batch1 = self.pw(x_pdp)
        batch1 = F.adaptive_avg_pool2d(batch1, (batch2.size(2), batch2.size(3)))
        
        x = torch.cat((batch1, batch2), dim=1)       
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
