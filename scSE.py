import torch.nn as nn
import torch

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1) # -> (B, C, 1, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        avg_pool = self.global_avg_pool(x) # -> (B, C, 1, 1)
        avg_pool = avg_pool.view(B, C)
        avg_pool = self.mlp(avg_pool) # -> (B, C)
        avg_pool = self.sigmoid(avg_pool).view(B, C, 1, 1) # (B, C, 1, 1)
        return avg_pool * x # (B, C, 1, 1) * (B, C, H, W) -> (B, C, H, W)


class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        spatial_weight = self.sigmoid(self.conv(x)) # -> (B, 1, H, W)
        return spatial_weight * x # (B, 1, H, W) * (B, C, H, W) -> (B, C, H, W)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)
    
    def forward(self, x):
        channel_out = self.cSE(x) # (B, C, H, W)
        spatial_out = self.sSE(x) # (B, C, H, W)
        return channel_out + spatial_out #  -> (B, C, H, W)

if __name__ == "__main__":
    x = torch.randn(1, 9, 96, 96)
    model = scSE(in_channels=9)
    print(model(x).shape)

    # cal params in million
    print(sum(p.numel() if p.requires_grad else 0 for p in model.parameters()) / 1000000)