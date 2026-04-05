import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SE_Attention import SE, Flatten


class ModuleNew(nn.Module):
    def __init__(self, in_channels, out_channels, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dw1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            stride=stride,
            dilation=1,
            padding=1,
            kernel_size=3,
        )
        self.dw2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            stride=stride,
            dilation=2,
            padding=2,
            kernel_size=3,
        )
        self.pw_out = nn.Conv2d(
            in_channels=2 * in_channels, out_channels=out_channels, kernel_size=1
        )
        self.se = SE(gate_channels=out_channels, reduction_ratio=16)

        self.s = stride

    def forward(self, x):
        x_ori = x
        x1 = F.relu(self.dw1(x))
        x2 = F.relu(self.dw2(x))
        x = torch.cat((x1, x2), dim=1)
        x = self.shuffle(x)
        x = F.relu(self.pw_out(x))
        x = self.se(x)
        if self.s == 1 and x.size() == x_ori.size():
            x = x_ori + x
        return x

    def shuffle(self, x):
        num_group = 2
        b, num_channels, height, width = x.data.size()
        group_channels = num_channels // num_group
        x = x.reshape(b, group_channels, num_group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, num_channels, height, width)
        return x


class PDPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(PDPBlock, self).__init__()

        self.pw1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            padding=0,
            stride=1,
            kernel_size=1,
            bias=False,
        )
        self.dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pw2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.feat_att = SE(gate_channels=out_channels)

    def forward(self, x):
        x = self.pw1(x)
        x = F.relu(self.dw(x))
        x = F.relu(self.pw2(x))

        x_attention = self.feat_att(x)
        x = torch.mul(x, x_attention)
        return x


class ModulePDP(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ModulePDP, self).__init__()

        self.in_channels = in_channels
        self.stride = stride
        self.out_channels = out_channels
        self.pdp = PDPBlock(
            in_channels=in_channels, out_channels=out_channels, stride=stride
        )

        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=stride,
            bias=False,
        )

    def forward(self, x):
        x_pdp = self.pdp(x)

        if not (self.stride == 1 and x.size() == x_pdp.size()):
            x = self.pw(x)

        x = torch.add(x, x_pdp)
        return F.relu(x)


class PDPNet(nn.Module):
    def __init__(self, in_channels, image_size, num_classes=5, *args, **kwargs):
        super(PDPNet, self).__init__(*args, **kwargs)

        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            ModuleNew(in_channels=32, out_channels=64, stride=1),
            ModuleNew(in_channels=64, out_channels=64, stride=1),
            ModuleNew(
                in_channels=64, out_channels=128, stride=1 if image_size == 32 else 2
            ),
            ModuleNew(in_channels=128, out_channels=128, stride=1),
            ModuleNew(in_channels=128, out_channels=256, stride=2),
            ModuleNew(in_channels=256, out_channels=256, stride=1),
            ModuleNew(in_channels=256, out_channels=256, stride=2),
            ModuleNew(in_channels=256, out_channels=512, stride=1),
            ModuleNew(in_channels=512, out_channels=512, stride=2),
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self, x):
        x = self.seq(x)
        return x
