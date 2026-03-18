
# Unet model

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()
        self.in_channels = in_channels # 1
        self.out_channels = out_channels #2

        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)

        self.conv_up4 = DoubleConv(1024, 512)
        self.conv_up3 = DoubleConv(512, 256)
        self.conv_up2 = DoubleConv(256, 128)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    def crop(self, skip, target):
        _, _, h, w = skip.shape
        _, _, th, tw = target.shape

        dh = (h - th) // 2
        dw = (w - tw) // 2

        return skip[:, :, dh:dh+th, dw:dw+tw]



    def forward(self, x):
        x1_skip = self.conv1(x) # (1, 572, 572) -> (64, 568, 568)
        x2 = self.down(x1_skip) # (64, 568, 568) -> (64, 284, 284)
        x2_skip = self.conv2(x2) # (64, 284, 284) -> (128, 280, 280)
        x3 = self.down(x2_skip) # (128, 280, 280) -> (128, 140, 140)
        x3_skip = self.conv3(x3) # (128, 140, 140) -> (256, 136, 136)
        x4 = self.down(x3_skip) # (256, 136, 136) -> (256, 68, 68)
        x4_skip = self.conv4(x4) # (256, 68, 68) -> (512, 64, 64)
        x5 = self.down(x4_skip) # (512, 64, 64) -> (512, 32, 32)
        x5 = self.conv5(x5) # (512, 32, 32) -> (1024, 28, 28)

        x4_up = self.up_conv4(x5) # (1024, 28, 28) -> (512, 56, 56)
        x4_skip_crop = self.crop(x4_skip, x4_up) # (512, 64, 64) -> (512, 56, 56)
        x4_up_concat = torch.cat([x4_skip_crop, x4_up], dim=1) # (512+512, 56, 56) -> (1024, 56, 56)
        x4_up_conv = self.conv_up4(x4_up_concat) # (1024, 56, 56) -> (512, 52, 52)

        x3_up = self.up_conv3(x4_up_conv) # (512, 52, 52) -> (256, 104, 104)
        x3_skip_crop = self.crop(x3_skip, x3_up) # (256, 136, 136) -> (256, 104, 104)
        x3_up_concat = torch.cat([x3_skip_crop, x3_up], dim=1) # (256+256, 104, 104) -> (512, 104, 104)
        x3_up_conv = self.conv_up3(x3_up_concat) # (512, 104, 104) -> (256, 100, 100)

        x2_up = self.up_conv2(x3_up_conv) # (256, 100, 100) -> (128, 200, 200)
        x2_skip_crop = self.crop(x2_skip, x2_up) # (128, 280, 280) -> (128, 200, 200)
        x2_up_concat = torch.cat([x2_skip_crop, x2_up], dim=1) # (128+128, 200, 200) -> (256, 200, 200)
        x2_up_conv = self.conv_up2(x2_up_concat) # (256, 200, 200) -> (128, 196, 196)

        x1_up = self.up_conv1(x2_up_conv) # (128, 196, 196) -> (64, 392, 392)
        x1_skip_crop = self.crop(x1_skip, x1_up) # (64, 568, 568) -> (64, 392, 392)
        x1_up_concat = torch.cat([x1_skip_crop, x1_up], dim=1) # (64+64, 392, 392) -> (128, 392, 392)
        x1_up_conv = self.conv_up1(x1_up_concat) # (128, 392, 392) -> (64, 388, 388)
        x1_out = self.conv_out(x1_up_conv) # (64, 388, 388) -> (2, 388, 388)

        return x1_out


if __name__ == "__main__":
    # original unet from scratch
    x = torch.randn(1, 1, 572, 572)
    model = Unet(in_channels=1, out_channels=2)
    print(model(x).shape)

    # cal params in million
    print(sum(p.numel() for p in model.parameters()) / 1000000)