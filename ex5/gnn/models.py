import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(256*4*4, 1)
        )

    def forward(self, x):
        return self.net(x).view(-1)


class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        # project & reshape into “256×4×4”
        self.fc = nn.Linear(latent_dim, 256*4*4)

        self.net = nn.Sequential(
            # 256×4×4 → 128×8×8
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 128×8×8 → 64×16×16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64×16×16 → 3×32×32
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()   # outputs in [–1,+1]
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 4, 4)
        return self.net(x)
