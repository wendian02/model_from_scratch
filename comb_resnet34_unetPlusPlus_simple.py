import torch
from torch import nn
from torchvision.models import resnet34
from scSE import scSE


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        backbone = resnet34(weights=None)

        # replace first conv if in_channels != 3
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # (64, 128, 128)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # (64, 64, 64)
        self.layer2 = backbone.layer2  # (128, 32, 32)
        self.layer3 = backbone.layer3  # (256, 16, 16)
        self.layer4 = backbone.layer4  # (512, 8, 8)

    def forward(self, x):
        x0_0 = self.layer0(x)                      # (64, 128, 128)
        x1_0 = self.layer1(self.maxpool(x0_0))     # (64, 64, 64)
        x2_0 = self.layer2(x1_0)                   # (128, 32, 32)
        x3_0 = self.layer3(x2_0)                   # (256, 16, 16)
        x4_0 = self.layer4(x3_0)                   # (512, 8, 8)
        return x0_0, x1_0, x2_0, x3_0, x4_0


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.attention1 = scSE(in_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.attention2 = scSE(out_channels)

    def forward(self, x):
        x = self.attention1(x)
        x = self.block(x)
        x = self.attention2(x)
        return x


class UnetPlusPlus(nn.Module):
    """
    Aligned with SMP UnetPlusPlus(encoder_name='resnet34', decoder_channels=(256,128,64,32,16)).

    Encoder features:
      feats[0]: (512,   8,   8)  -- bottleneck (layer4)
      feats[1]: (256,  16,  16)  -- layer3
      feats[2]: (128,  32,  32)  -- layer2
      feats[3]: (64,   64,  64)  -- layer1
      feats[4]: (64,  128, 128)  -- layer0

    SMP decoder block: DecoderBlock(x, skip) = up(x) -> cat(up_x, skip) -> attn1 -> conv1 -> conv2 -> attn2
    SMP node naming: x_{depth_idx}_{layer_idx}
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.resnet34 = ResNet34Encoder(in_channels)

        # x_0_0: up(f512)+f256 = 768 -> 256
        self.x_0_0 = DoubleConv(512+256, 256)

        # x_1_1: up(f256)+f128 = 384 -> 128
        self.x_1_1 = DoubleConv(256+128, 128)
        # x_0_1: up(x_0_0)+cat(x_1_1,f128) = 256+128+128 = 512 -> 128
        self.x_0_1 = DoubleConv(256+128+128, 128)

        # x_2_2: up(f128)+f64 = 192 -> 64
        self.x_2_2 = DoubleConv(128+64, 64)
        # x_1_2: up(x_1_1)+cat(x_2_2,f64) = 128+64+64 = 256 -> 64
        self.x_1_2 = DoubleConv(128+64+64, 64)
        # x_0_2: up(x_0_1)+cat(x_1_2,x_2_2,f64) = 128+64+64+64 = 320 -> 64
        self.x_0_2 = DoubleConv(128+64+64+64, 64)

        # x_3_3: up(f64)+f64_layer0 = 128 -> 64
        self.x_3_3 = DoubleConv(64+64, 64)
        # x_2_3: up(x_2_2)+cat(x_3_3,f64_layer0) = 64+64+64 = 192 -> 64
        self.x_2_3 = DoubleConv(64+64+64, 64)
        # x_1_3: up(x_1_2)+cat(x_2_3,x_3_3,f64_layer0) = 64+64+64+64 = 256 -> 64
        self.x_1_3 = DoubleConv(64+64+64+64, 64)
        # x_0_3: up(x_0_2)+cat(x_1_3,x_2_3,x_3_3,f64_layer0) = 64+64+64+64+64 = 320 -> 32
        self.x_0_3 = DoubleConv(64+64+64+64+64, 32)

        # x_0_4: up(x_0_3), no skip = 32 -> 16
        self.x_0_4 = DoubleConv(32, 16)

        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        x0_0, x1_0, x2_0, x3_0, x4_0 = self.resnet34(x)

        # feats (reversed, skip first removed): [f512, f256, f128, f64_layer1, f64_layer0]
        f0, f1, f2, f3, f4 = x4_0, x3_0, x2_0, x1_0, x0_0

        # --- diagonal (layer_idx=0) ---
        d_0_0 = self.x_0_0(torch.cat([self.up(f0), f1], 1))   # (256, 16, 16)
        d_1_1 = self.x_1_1(torch.cat([self.up(f1), f2], 1))   # (128, 32, 32)
        d_2_2 = self.x_2_2(torch.cat([self.up(f2), f3], 1))   # (64,  64, 64)
        d_3_3 = self.x_3_3(torch.cat([self.up(f3), f4], 1))   # (64, 128,128)

        # --- layer_idx=1 ---
        d_0_1 = self.x_0_1(torch.cat([self.up(d_0_0), d_1_1, f2], 1))   # (128, 32, 32)
        d_1_2 = self.x_1_2(torch.cat([self.up(d_1_1), d_2_2, f3], 1))   # (64,  64, 64)
        d_2_3 = self.x_2_3(torch.cat([self.up(d_2_2), d_3_3, f4], 1))   # (64, 128,128)

        # --- layer_idx=2 ---
        d_0_2 = self.x_0_2(torch.cat([self.up(d_0_1), d_1_2, d_2_2, f3], 1))   # (64, 64, 64)
        d_1_3 = self.x_1_3(torch.cat([self.up(d_1_2), d_2_3, d_3_3, f4], 1))   # (64, 128,128)

        # --- layer_idx=3 ---
        d_0_3 = self.x_0_3(torch.cat([self.up(d_0_2), d_1_3, d_2_3, d_3_3, f4], 1))  # (32, 128,128)

        # --- final ---
        d_0_4 = self.x_0_4(self.up(d_0_3))   # (16, 256, 256)

        return self.conv_out(d_0_4)



if __name__ == "__main__":
    model = UnetPlusPlus(in_channels=11, out_channels=1)
    x = torch.randn(1, 11, 256, 256)
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()) / 1000000)