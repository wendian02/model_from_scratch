import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

img_rows = 28
img_cols = 28
channels = 1
img_shape = (channels, img_rows, img_cols)
latent_dim = 100


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.Linear(1024, int(np.prod(img_shape))), # channels x img_rows x img_cols
            nn.Tanh(), # -1 to 1
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *img_shape)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        return self.model(img)


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


generator = Generator().to(device)
discriminator = Discriminator().to(device)
torch.compile(generator)
torch.compile(discriminator)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)


def save_imgs(epoch):
    r, c = 5, 5
    z = torch.randn(r * c, latent_dim, device=device)
    generator.eval()
    with torch.no_grad():
        gen_imgs = generator(z).cpu().numpy()
    generator.train()

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, 0, :, :], cmap="gray")
            axs[i, j].axis("off")
            cnt += 1
    fig.savefig("images/mnist_%d.png" % epoch)
    plt.close()


def train(epochs, batch_size=128, save_interval=50):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for imgs, _ in dataloader:
            batch = imgs.size(0)
            real = torch.ones(batch, 1, device=device) # B, 1
            fake = torch.zeros(batch, 1, device=device)

            imgs = imgs.to(device)

            z = torch.randn(batch, latent_dim, device=device)
            gen_imgs = generator(z)

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), real)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            g_loss = adversarial_loss(discriminator(gen_imgs), real)
            g_loss.backward()
            optimizer_G.step()

        print(f"{epoch} [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        if epoch % save_interval == 0:
            save_imgs(epoch)


train(epochs=100, batch_size=32, save_interval=10)

torch.save(generator.state_dict(), "generator_model.pth")
