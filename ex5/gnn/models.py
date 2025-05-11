import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(
                128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(
                256), nn.LeakyReLU(0.2, True),
            nn.Flatten()
        )
        self.adv_head = nn.Linear(256*4*4, 1)       # real/fake
        self.aux_head = nn.Linear(256*4*4, n_classes)  # class logits

    def forward(self, x):
        h = self.features(x)                    # [B, 256*4*4]
        real_fake = self.adv_head(h).view(-1)    # [B]
        class_logits = self.aux_head(h)          # [B, n_classes]
        return real_fake, class_logits


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        # embed label into same dim as z
        self.label_emb = nn.Embedding(n_classes, latent_dim)
        # project combined (z + label) into 256×4×4
        self.fc = nn.Linear(latent_dim * 2, 256*4*4)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(
                128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.BatchNorm2d(
                64),  nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),    nn.Tanh()   # → [–1,+1]
        )

    def forward(self, z, y):
        # z: [B, latent_dim], y: [B] int labels
        y_vec = self.label_emb(y)                 # → [B, latent_dim]
        x = torch.cat([z, y_vec], dim=1)          # → [B, latent_dim*2]
        x = self.fc(x).view(-1, 256, 4, 4)         # → [B,256,4,4]
        return self.net(x)                        # → [B,3,32,32]


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_channels, latent_dim, img_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size

        # === ENCODER ===
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),  # 32x -> 16x
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 16x -> 8x
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 8x -> 4x
            nn.ReLU(True),
            nn.Flatten() # [B, 128, 4, 4] -> [B, 128*4*4] = [B, 2048]
        )

        # Layers fully connected 
        self.fc_mu = nn.Linear(128 * (self.img_size // 8)**2, latent_dim)
        self.fc_logvar = nn.Linear(128 * (self.img_size // 8)**2, latent_dim)

        # === DECODER ===
        self.decoder_fc_input_size = 128 * (self.img_size // 8)**2
        self.decoder_fc = nn.Linear(latent_dim, self.decoder_fc_input_size)

        # Convolutional layers
        self.decoder_conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> [B, 64, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [B, 32, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # -> [B, input, 32, 32]
            nn.Sigmoid()
        )

    def encode(self, x) -> tuple(torch.Tensor, torch.Tensor):
        h = self.encoder_conv(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, self.img_size // 8, self.img_size // 8)
        return self.decoder_conv_transpose(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
