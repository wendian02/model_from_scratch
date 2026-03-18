import torch
from torch import nn
from scSE import scSE
from torchvision.models import resnet34

class ResNet34(nn.Module):
    def __init__(self, in_channels, out_channels):
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, out_channels)

    
    def forward(self, x):
        x0_0 = self.layer0(x)
        x1_0 = self.layer1(self.maxpool(x0_0))
        x2_0 = self.layer2(x1_0)
        x3_0 = self.layer3(x2_0)
        x4_0 = self.layer4(x3_0)
        x = self.avgpool(x4_0)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = ResNet34(in_channels=3, out_channels=1000)
    x = torch.randn(1, 3, 256, 256)
    print(model(x).shape)

    print(sum(p.numel() for p in model.parameters()) / 1000000)

    from torchinfo import summary
    summary(model, input_size=(1, 3, 256, 256))

    # torch.onnx.export(model, x, "resnet34.onnx")
