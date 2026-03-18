import torch
from torch import nn


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, use_1x1conv=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if use_1x1conv:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))
        else:
            self.downsample = None

        self.use_1x1conv = use_1x1conv

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_1x1conv:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # stage 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # stage 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2 = self._make_layer(64, 64, 3, is_first=True)

        self.stage3 = self._make_layer(64, 128, 4) # stage 3
        self.stage4 = self._make_layer(128, 256, 6) # stage 4
        self.stage5 = self._make_layer(256, 512, 3) # stage 5

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_channels)

    def _make_layer(self, in_channels, out_channels, blocks, is_first=False):
        layers = []
        for i in range(blocks):
            if i == 0 and not is_first:
                layers.append(BasicBlock(in_channels, out_channels, stride=2, use_1x1conv=True))
            else:
                layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x) # (1, 256, 256) -> (64, 128, 128)
        x = self.bn1(x)
        x = self.relu(x)

        # stage 2
        x = self.maxpool(x) # (64, 128, 128) -> (64, 64, 64)
        x = self.stage2(x) # (64, 64, 64) -> (64, 64, 64)

        x = self.stage3(x) # (64, 64, 64) -> (128, 32, 32)
        x = self.stage4(x) # (128, 32, 32) -> (256, 16, 16)
        x = self.stage5(x) # (256, 16, 16) -> (512, 8, 8)
        x = self.avgpool(x) # (512, 8, 8) -> (512, 1, 1)
        x = x.view(x.size(0), -1) # (512, 1, 1) -> (512,)
        x = self.fc(x) # (512,) -> (1000,)
        return x

if __name__ == "__main__":
    # this script is the same as the torchvision.models.resnet34
    model = ResNet34(in_channels=3, out_channels=1000)
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)

    print(sum(p.numel() for p in model.parameters()) / 1000000)

    from torchinfo import summary
    summary(model, input_size=(1, 3, 256, 256))



