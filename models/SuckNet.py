import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PDPNet import PDPBlock, CGDF

class SuckNet(nn.Module):
    def __init__(self, image_size, num_classes, *args, **kwargs):
        super(SuckNet, self).__init__()
        
        self.init_seq = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
		)
        
        self.branch1 = nn.Sequential(
			PDPBlock(in_channels=64, out_channels=128, stride=1),
			PDPBlock(in_channels=128, out_channels=128, stride=1),
			PDPBlock(in_channels=128, out_channels=128, stride=2 if image_size == 32 else 1),
			PDPBlock(in_channels=128, out_channels=256, stride=1),
			PDPBlock(in_channels=256, out_channels=256, stride=2),
			PDPBlock(in_channels=256, out_channels=512, stride=1),
			PDPBlock(in_channels=512, out_channels=512, stride=2)
		)
        
        self.branch2 = nn.Sequential(
			CGDF(in_channels=64, out_channels=128, stride=1),
			CGDF(in_channels=128, out_channels=128, stride=1),
			CGDF(in_channels=128, out_channels=128, stride=2 if image_size == 32 else 1),
			CGDF(in_channels=128, out_channels=256, stride=1),
			CGDF(in_channels=256, out_channels=256, stride=2),
			CGDF(in_channels=256, out_channels=512, stride=1),
			CGDF(in_channels=512, out_channels=512, stride=2)
		)
        
        self.pw = nn.Conv2d(1024, 1024, kernel_size=1)
        
        self.head = nn.Sequential(
			nn.AdaptiveAvgPool2d((1, 1)),
			nn.Flatten(),
			nn.Linear(1024, num_classes)
		)
        
    def forward(self, x):
        x = self.init_seq(x)
        
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        
        x = torch.cat((x1, x2), dim=1)
        x = F.relu(self.pw(x))
        
        x = self.head(x)
        
        return x