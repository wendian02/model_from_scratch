#%%
from torchvision.datasets import OxfordIIITPet
import matplotlib.pyplot as plt
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image

from einops import rearrange

from einops.layers.torch import Rearrange
from torch import Tensor
import torch
import torch.nn as nn

from torchvision.transforms import Normalize



to_tensor = [Resize((144, 144)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image = t(image)
        return image, target

def show_images(images, num_samples=40, cols=8):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    idx = int(len(dataset) / num_samples)
    print(images)
    for i, img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples/cols) + 1, cols, int(i/idx) + 1)
            plt.imshow(to_pil_image(img[0]))

# 200 images for each pet
dataset = OxfordIIITPet(root="./data", download=True, transforms=Compose(to_tensor))
# show_images(dataset)



class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 8, emb_size = 128):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim,num_heads=n_heads,dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        _x = x
        x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x = self.attn_dropout(x)
        x = x + _x

        _x = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + _x
        return x

transformer_encoder = TransformerEncoder(dim=128, n_heads=4, dropout=0.1)
x = torch.randn(1, 10, 128)
x = transformer_encoder(x)
print(x.shape)

#%%
from einops import repeat

class ViT(nn.Module):
    def __init__(self, ch=3, img_size=144, patch_size=4, emb_dim=32,
                n_layers=6, out_dim=37, dropout=0.1, heads=2):
        super().__init__()

        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.n_layers = n_layers

        self.patch_embedding = PatchEmbedding(in_channels=ch,patch_size=patch_size, emb_size=emb_dim)

        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))

        self.emb_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = TransformerEncoder(dim=emb_dim, n_heads=heads, dropout=dropout)
            self.layers.append(transformer_block)
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))


    def forward(self, img):
        x = self.patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x)
        return self.head(x[:, 0, :])


model = ViT()
print(model)
y = model(torch.ones((1, 3, 144, 144)))

#%%

from torch.utils.data import DataLoader
from torch.utils.data import random_split

train_split = int(0.8 * len(dataset))
train, test = random_split(dataset, [train_split, len(dataset) - train_split])

train_dataloader = DataLoader(train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test, batch_size=32, shuffle=False)
import torch.optim as optim
import numpy as np

device = "cuda"
model = ViT().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
criterion = nn.CrossEntropyLoss()

for epoch in range(1000):
    epoch_losses = []
    model.train()
    for step, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_losses.append(loss.item())
    scheduler.step()
    if epoch % 5 == 0:
        print(f">>> Epoch {epoch} train loss: ", np.mean(epoch_losses))
        epoch_losses = []
        model.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(test_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_losses.append(loss.item())
        print(f">>> Epoch {epoch} test loss: ", np.mean(epoch_losses))

#%%
inputs, labels = next(iter(test_dataloader))
inputs, labels = inputs.to(device), labels.to(device)
outputs = model(inputs)

print("Predicted classes", outputs.argmax(-1))
print("Actual classes", labels)