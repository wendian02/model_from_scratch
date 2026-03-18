import torch
from torch import nn
from scSE import scSE
from torchvision.models import resnet34


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        backbone = resnet34(weights=None)
        # replace first conv if in_channels != 3
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
    
    def forward(self, x):
        x0_0 = self.layer0(x)
        x1_0 = self.layer1(self.maxpool(x0_0))
        x2_0 = self.layer2(x1_0)
        x3_0 = self.layer3(x2_0)
        x4_0 = self.layer4(x3_0)
        return x0_0, x1_0, x2_0, x3_0, x4_0


class BasicBlock(nn.Module):
    expansion: int = 1

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
    def __init__(self, in_channels):
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

    def _make_layer(self, in_channels, out_channels, blocks, is_first=False):
        layers = []
        for i in range(blocks):
            if i == 0 and not is_first:
                layers.append(BasicBlock(in_channels, out_channels, stride=2, use_1x1conv=True))
            else:
                layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # stage 1
        x = self.conv1(x) # (1, 256, 256) -> (64, 128, 128)
        x = self.bn1(x)
        x0_0 = self.relu(x) # (64, 128, 128)

        # stage 2
        x = self.maxpool(x0_0) # (64, 128, 128) -> (64, 64, 64)
        x1_0 = self.stage2(x) # (64, 64, 64) -> (64, 64, 64)

        x2_0 = self.stage3(x1_0) # (64, 64, 64) -> (128, 32, 32)
        x3_0 = self.stage4(x2_0) # (128, 32, 32) -> (256, 16, 16)
        x4_0 = self.stage5(x3_0) # (256, 16, 16) -> (512, 8, 8)
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels # 1
        self.out_channels = out_channels #2
        # self.resnet34 = ResNet34Encoder(in_channels)

        self.resnet34 = ResNet34(in_channels)

        self.conv0_1 = DoubleConv(64+64, 64)

        self.conv1_1 = DoubleConv(64+128, 64)
        self.conv0_2 = DoubleConv(64+64+64, 64)

        self.conv2_1 = DoubleConv(128+256, 128)
        self.conv1_2 = DoubleConv(64+64+128, 64)
        self.conv0_3 = DoubleConv(64+64+64+64, 64)

        self.conv3_1 = DoubleConv(256+512, 256)
        self.conv2_2 = DoubleConv(128+128+256, 128)
        self.conv1_3 = DoubleConv(64+64+64+128, 64)
        self.conv0_4 = DoubleConv(64+64+64+64+64, 32)
        self.conv_second_to_last = DoubleConv(32, 16)

        self.conv_out = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):

        # x0_0: (64, 128, 128)
        # x1_0: (64, 64, 64) 
        # x2_0: (128, 32, 32)
        # x3_0: (256, 16, 16)
        # x4_0: (512, 8, 8)

        x0_0, x1_0, x2_0, x3_0, x4_0 = self.resnet34(x)

        x0_1 = self.up(x1_0) # (64, 64, 64) -> (64, 128, 128)
        x0_1 = torch.cat([x0_1, x0_0], dim=1) # (64, 128, 128) -> (64+64, 128, 128)
        x0_1 = self.conv0_1(x0_1) # (64+64, 128, 128) -> (64, 128, 128)


        x1_1 = self.up(x2_0) # (128, 32, 32) -> (128, 64, 64)
        x1_1 = torch.cat([x1_1, x1_0], dim=1) # (128, 64, 64) -> (64+128, 64, 64)
        x1_1 = self.conv1_1(x1_1) # (64+128, 64, 64) -> (64, 64, 64)
        x0_2 = self.up(x1_1) # (64, 64, 64) -> (64, 128, 128)
        x0_2 = torch.cat([x0_2, x0_1, x0_0], dim=1) # (64, 128, 128) -> (64+64+64, 128, 128)
        x0_2 = self.conv0_2(x0_2) # (64+64+64, 128, 128) -> (64, 128, 128)


        x2_1 = self.up(x3_0) # (256, 16, 16) -> (256, 32, 32)
        x2_1 = torch.cat([x2_1,x2_0], dim=1) # (256, 32, 32) -> (128+256, 32, 32)
        x2_1 = self.conv2_1(x2_1) # (128+256, 32, 32) -> (128, 32, 32)
        x1_2 = self.up(x2_1) # (128, 32, 32) -> (128, 64, 64)
        x1_2 = torch.cat([x1_2, x1_1, x1_0], dim=1) # (128, 64, 64)-> (64+64+128, 64, 64)
        x1_2 = self.conv1_2(x1_2) # (64+64+128, 64, 64) -> (64, 64, 64)
        x0_3 = self.up(x1_2) # (64, 64, 64) -> (64, 128, 128)
        x0_3 = torch.cat([x0_3, x0_2, x0_1, x0_0], dim=1) # (64, 128, 128)-> (64+64+64+64, 128, 128)
        x0_3 = self.conv0_3(x0_3) # (64+64+64+64, 128, 128) -> (64, 128, 128)


        x3_1 = self.up(x4_0) # (512, 8, 8) -> (512, 16, 16)
        x3_1 = torch.cat([x3_1,x3_0], dim=1) # (512, 16, 16) -> (256+512, 16, 16)
        x3_1 = self.conv3_1(x3_1) # (256+512, 16, 16) -> (256, 16, 16)
        x2_2 = self.up(x3_1) # (256, 16, 16) -> (256, 32, 32)
        x2_2 = torch.cat([x2_2, x2_1, x2_0], dim=1) # (256, 32, 32)-> (128+128+256, 32, 32)
        x2_2 = self.conv2_2(x2_2) # (128+128+256, 32, 32) -> (128, 32, 32)
        x1_3 = self.up(x2_2) # (128, 32, 32) -> (128, 64, 64)
        x1_3 = torch.cat([x1_3, x1_2, x1_1, x1_0], dim=1) # (128, 64, 64)-> (64+64+64+128, 64, 64)
        x1_3 = self.conv1_3(x1_3) # (64+64+64+128, 64, 64) -> (64, 64, 64)
        x0_4 = self.up(x1_3) # (64, 64, 64) -> (64, 128, 128)
        x0_4 = torch.cat([x0_4, x0_3, x0_2, x0_1, x0_0], dim=1) # (64, 128, 128) -> (64+64+64+64+64, 128, 128)
        x0_4 = self.conv0_4(x0_4) # (64+64+64+64+64, 128, 128) -> (32, 128, 128)

        second_to_last = self.up(x0_4)                               # (32, 128, 128) -> (32, 256, 256)
        second_to_last = self.conv_second_to_last(second_to_last) # (32, 256, 256) -> (16, 256, 256)
        x_out = self.conv_out(second_to_last)                         # -> (out_ch, 256, 256)

        return x_out



if __name__ == "__main__":
    "scratch model = smp model"
    model = UnetPlusPlus(in_channels=11, out_channels=1)
    x = torch.randn(1, 11, 256, 256)
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()) / 1000000)
    # from torchinfo import summary
    # summary(model, input_size=(1, 11, 256, 256))
    # state_dict = model.state_dict()
    # print(state_dict.keys())