import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        self.down1 = ConvBlock(3, 64)          # down1.conv.*
        self.down2 = ConvBlock(64, 128)        # down2.conv.*
        self.bridge = ConvBlock(128, 256)      # bridge.conv.*

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # up2.*
        self.upconv2 = ConvBlock(256, 128)      # upconv2.conv.*

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # up1.*
        self.upconv1 = ConvBlock(128, 64)       # upconv1.conv.*

        self.out = nn.Conv2d(64, 1, kernel_size=1)  # out.*

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(F.max_pool2d(x1, 2))
        x3 = self.bridge(F.max_pool2d(x2, 2))

        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv2(x)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv1(x)

        x = self.out(x)
        return x

UNet = LaneNet

