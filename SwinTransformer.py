import torch.nn as nn
import torch
from utils import SwinEmbedding, PatchMerging, SwinEncoder

class Swin(nn.Module):
    def __init__(self):
        super().__init__()
        self.Embedding = SwinEmbedding()
        self.PatchMerging = nn.ModuleList()
        emb_size = 96
        num_class = 5
        for i in range(3):
            self.PatchMerging.append(PatchMerging(emb_size))
            emb_size *= 2
        
        self.stage1 = SwinEncoder(96, 3)
        self.stage2 = SwinEncoder(192, 6)
        self.stage3 = nn.ModuleList([SwinEncoder(384, 12),
                                     SwinEncoder(384, 12),
                                     SwinEncoder(384, 12) 
                                    ])
        self.stage4 = SwinEncoder(768, 24)
        
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size = 1)
        
        self.layer = nn.Linear(768, num_class)

    def forward(self, x):
        x = self.Embedding(x)
        x = self.stage1(x)
        x = self.PatchMerging[0](x)
        x = self.stage2(x)
        x = self.PatchMerging[1](x)
        for stage in self.stage3:
            x = stage(x)
        x = self.PatchMerging[2](x)
        x = self.stage4(x)
        x = self.layer(self.avgpool1d(x.transpose(1, 2)).squeeze(2))
        return x

if __name__ == '__main__':
    # Usage Example (assuming num_classes = 5)

    x = torch.rand(1, 3, 224, 224)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.type(torch.FloatTensor).to(device)
    model = Swin().to(device)
    print(model(x).shape) 