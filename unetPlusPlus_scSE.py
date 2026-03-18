

import torch
import torch.nn as nn
import torch.nn.functional as F
from scSE import scSE

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # modern
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # modern
            nn.ReLU(),
        )
        self.scSE = scSE(out_channels) # scSE block after conv
    def forward(self, x):
        return self.scSE(self.block(x))


class UnetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels # 1
        self.out_channels = out_channels #2

        self.conv0_0 = DoubleConv(in_channels, 64)

        self.conv1_0 = DoubleConv(64, 128)
        self.conv0_1 = DoubleConv(192, 64)

        self.conv2_0 = DoubleConv(128, 256)
        self.conv1_1 = DoubleConv(384, 128)
        self.conv0_2 = DoubleConv(256, 64)

        self.conv3_0 = DoubleConv(256, 512)
        self.conv2_1 = DoubleConv(256+512, 256)
        self.conv1_2 = DoubleConv(128+128+256, 128)
        self.conv0_3 = DoubleConv(64+64+64+128, 64)

        self.conv4_0 = DoubleConv(512, 1024)
        self.conv3_1 = DoubleConv(512+1024, 512)
        self.conv2_2 = DoubleConv(256+256+512, 256)
        self.conv1_3 = DoubleConv(128+128+128+256, 128)
        self.conv0_4 = DoubleConv(64+64+64+64+128, 64)

        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1)


        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)



    def forward(self, x):
        x0_0 = self.conv0_0(x) # (1, 96, 96) -> (64, 96, 96)

        x1_0 = self.down(x0_0) # (64, 96, 96) -> (64, 48, 48)
        x1_0 = self.conv1_0(x1_0) # (64, 48, 48) -> (128, 48, 48)

        x0_1 = self.up(x1_0) # (128, 48, 48) -> (128, 96, 96)
        x0_1 = torch.cat([x0_0, x0_1], dim=1) # (128, 96, 96) -> (64+128, 96, 96)
        x0_1 = self.conv0_1(x0_1) # (192, 96, 96) -> (64, 96, 96)

        x2_0 = self.down(x1_0) # (128, 48, 48) -> (128, 24, 24)
        x2_0 = self.conv2_0(x2_0) # (128, 24, 24) -> (256, 24, 24)

        x1_1 = self.up(x2_0) # (256, 24, 24) -> (256, 48, 48)
        x1_1 = torch.cat([x1_0, x1_1], dim=1) # (256, 48, 48) -> (128+256, 48, 48)
        x1_1 = self.conv1_1(x1_1) # (256+128, 48, 48) -> (128, 48, 48)

        x0_2 = self.up(x1_1) # (128, 48, 48) -> (128, 96, 96)
        x0_2 = torch.cat([x0_0, x0_1, x0_2], dim=1) # (128, 96, 96) -> (64+64+128, 96, 96)
        x0_2 = self.conv0_2(x0_2) # (256, 96, 96) -> (64, 96, 96)

        x3_0 = self.down(x2_0) # (256, 24, 24) -> (256, 12, 12)
        x3_0 = self.conv3_0(x3_0) # (256, 12, 12) -> (512, 12, 12)
        x2_1 = self.up(x3_0) # (512, 12, 12) -> (512, 24, 24)
        x2_1 = torch.cat([x2_0, x2_1], dim=1) # (256, 24, 24) -> (256+512, 24, 24)
        x2_1 = self.conv2_1(x2_1) # (256+512, 24, 24) -> (256, 24, 24)
        x1_2 = self.up(x2_1) # (256, 24, 24) -> (, 48, 48)
        x1_2 = torch.cat([x1_0, x1_1, x1_2], dim=1) # (256, 48, 48) -> (128+128+256, 48, 48)
        x1_2 = self.conv1_2(x1_2) # (128+128+256, 48, 48) -> (128, 48, 48)
        x0_3 = self.up(x1_2) # (128, 48, 48) -> (128, 96, 96)
        x0_3 = torch.cat([x0_0, x0_1, x0_2, x0_3], dim=1) # (128, 96, 96) -> (64+64+64+128, 96, 96)
        x0_3 = self.conv0_3(x0_3) # (64+64+64+128, 96, 96) -> (64, 96, 96)

        x4_0 = self.down(x3_0) # (512, 12, 12) -> (512, 6, 6)
        x4_0 = self.conv4_0(x4_0) # (512, 6, 6) -> (1024, 6, 6)
        x3_1 = self.up(x4_0) # (1024, 6, 6) -> (1024, 12, 12)
        x3_1 = torch.cat([x3_0, x3_1], dim=1) # (512, 12, 12) -> (512+1024, 12, 12)
        x3_1 = self.conv3_1(x3_1) # (512+1024, 12, 12) -> (512, 12, 12)
        x2_2 = self.up(x3_1) # (512, 12, 12) -> (512, 24, 24)
        x2_2 = torch.cat([x2_0, x2_1, x2_2], dim=1) # (256, 24, 24) -> (256+256+512, 24, 24)
        x2_2 = self.conv2_2(x2_2) # (256+256+512, 24, 24) -> (256, 24, 24)
        x1_3 = self.up(x2_2) # (256, 24, 24) -> (256, 48, 48)
        x1_3 = torch.cat([x1_0, x1_1, x1_2, x1_3], dim=1) # (256, 48, 48) -> (128+128+128+256, 48, 48)
        x1_3 = self.conv1_3(x1_3) # (128+128+128+256, 48, 48) -> (128, 48, 48)
        x0_4 = self.up(x1_3) # (128, 48, 48) -> (128, 96, 96)
        x0_4 = torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4], dim=1) # (128, 96, 96) -> (64+64+64+64+128, 96, 96)
        x0_4 = self.conv0_4(x0_4) # (64+64+64+64+128, 96, 96) -> (64, 96, 96)

        x_out = self.conv_out(x0_4) # (64, 96, 96) -> (1, 96, 96)

        return x_out


if __name__ == "__main__":
    "encoder channels = decoder channels = (64, 128, 256, 512, 1024)"
    x = torch.randn(1, 9, 96, 96)
    model = UnetPlusPlus(in_channels=9, out_channels=1)
    print(model(x).shape)

    # cal params in million
    print(sum(p.numel() for p in model.parameters()) / 1000000)