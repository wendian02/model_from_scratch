import torch.nn as nn
import torch

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1) # global max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # global avg pooling

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        max_pool = self.max_pool(x) # (B, C, 1, 1)
        avg_pool = self.avg_pool(x) # (B, C, 1, 1)

        max_pool = max_pool.view(B, C)
        avg_pool = avg_pool.view(B, C)

        max_pool = self.mlp(max_pool)
        avg_pool = self.mlp(avg_pool)

        return self.sigmoid(max_pool + avg_pool).view(B, C, 1, 1) # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1)//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_layer, _ = x.max(dim=1, keepdim=True) # max_layer (B, 1, H, W)
        avg_layer = x.mean(dim=1, keepdim=True) # (B, 1, H, W)

        x = torch.cat([max_layer, avg_layer], dim=1) # (B, 2, H, W)
        x = self.conv(x) # (B, 1, H, W)

        return self.sigmoid(x) # (B, 1, H, W)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x) * x # (B, C, 1, 1) * (B, C, H, W) -> (B, C, H, W)
        x = self.spatial_attention(x) * x # (B, 1, H, W) * (B, C, H, W) -> (B, C, H, W)
        return x

if __name__ == "__main__":
    x = torch.randn(1, 9, 96, 96)
    model = CBAM(in_channels=9)
    print(model(x).shape)

    # cal params in million
    print(sum(p.numel() if p.requires_grad else 0 for p in model.parameters()) / 1000000)