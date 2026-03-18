import torch.nn as nn
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class get_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_model = smp.UnetPlusPlus(
            encoder_name="resnet34",  # 可选择多种backbone: resnet34, efficientnet-b0等
            encoder_weights=None,  # 使用预训练权重
            in_channels=11,  # S2 11 bands
            classes=1,  # Depth output
            activation=None,
            decoder_attention_type="scse",  # 使用空间和通道注意力机制
        )

    def forward(self, x):
        return self.depth_model(x)

if __name__ == "__main__":
    model = get_model()
    x = torch.randn(1, 11, 256, 256)
    print(model(x).shape)
    print(sum(p.numel() for p in model.parameters()) / 1000000)
    # from torchinfo import summary
    # summary(model, input_size=(1, 11, 256, 256))
    # state_dict = model.state_dict()
    # print(state_dict.keys())