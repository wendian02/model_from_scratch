import torch
import torch.nn as nn
import timm


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), #  H, W -> 2*H, 2*W
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1), # C -> C/2
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1) # concat skip
        return self.conv(x)
 

class UNetSwinT(nn.Module):
    def __init__(self, num_classes=1, in_channels=3):
        super().__init__()

        # Encoder: swin_base 4 stages, returns NHWC tensors
        # s1: (B, H/4,  W/4,  128)
        # s2: (B, H/8,  W/8,  256)
        # s3: (B, H/16, W/16, 512)
        # s4: (B, H/32, W/32, 1024)
        self.encoder = timm.create_model(
            'swin_base_patch4_window7_224',
            img_size=128,
            window_size=4,    # 128/4(patch)=32, 32/4(window)=8 ✓
            in_chans=in_channels,
            pretrained=False,
            features_only=True,
        )

        # Stem: CNN provides H and H/2 skip connections
        self.stem0 = ConvBlock(in_channels, 32)                          # H,   32ch  → s0
        self.stem1 = nn.Sequential(                                      # H/2, 64ch  → s_half
            nn.Conv2d(32, 64, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ConvBlock(64, 64),
        )

        # Bottleneck (stage 5): CNN at H/32
        self.bottleneck = ConvBlock(1024, 1024)

        # Decoder
        self.up4 = UpBlock(1024, 512, 512)   # H/32 → H/16, skip s3
        self.up3 = UpBlock(512,  256, 256)   # H/16 → H/8,  skip s2
        self.up2 = UpBlock(256,  128, 128)   # H/8  → H/4,  skip s1
        self.up1 = UpBlock(128,  64,  64)    # H/4  → H/2,  skip stem1
        self.up0 = UpBlock(64,   32,  32)    # H/2  → H,    skip stem0
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Stem skip connections
        s0     = self.stem0(x)               # (B, 32,  H,    W)
        s_half = self.stem1(s0)              # (B, 64,  H/2,  W/2)

        # Swin encoder — timm returns NHWC → permute to NCHW
        s1, s2, s3, s4 = [f.permute(0, 3, 1, 2) for f in self.encoder(x)]

        # Bottleneck
        x = self.bottleneck(s4)

        # Decoder
        x = self.up4(x, s3)       # (B, 512, H/16, W/16)
        x = self.up3(x, s2)       # (B, 256, H/8,  W/8)
        x = self.up2(x, s1)       # (B, 128, H/4,  W/4)
        x = self.up1(x, s_half)   # (B, 64,  H/2,  W/2)
        x = self.up0(x, s0)       # (B, 32,  H,    W)
        return self.head(x)       # (B, num_classes, H, W)


if __name__ == '__main__':
    x = torch.rand(1, 10, 128, 128)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.to(device)
    model = UNetSwinT(num_classes=1, in_channels=10).to(device)
    print(model(x).shape)  # expected: torch.Size([1, 1, 128, 128])
