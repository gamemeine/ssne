import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            # 3×32×32 → 128×16×16
            nn.Conv2d(3,   128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 128×16×16 → 256×8×8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 256×8×8 → 512×4×4
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()  # → [B, 512*4*4]
        )
        feat_dim = 512 * 4 * 4

        # real/fake head
        self.adv_head = nn.Linear(feat_dim, 1)
        # label embedding for projection
        self.label_emb = nn.Embedding(n_classes, feat_dim)

    def forward(self, x, y):
        h = self.features(x)                    # [B, feat_dim]
        real_logit = self.adv_head(h).view(-1)   # [B]
        v_y = self.label_emb(y)          # [B, feat_dim]
        proj = torch.sum(h * v_y, dim=1)   # [B]
        return real_logit + proj                # [B]


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(n_classes, latent_dim)

        # project z and y → 512×4×4
        self.noise_proj = nn.Sequential(
            nn.Linear(latent_dim, 512*4*4),
            nn.ReLU(True)
        )
        self.label_proj = nn.Sequential(
            nn.Linear(latent_dim, 512*4*4),
            nn.ReLU(True)
        )

        self.net = nn.Sequential(
            # 1024×4×4 → 512×8×8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 512×8×8 → 256×16×16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 256×16×16 → 128×32×32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # finalize to RGB
            nn.Conv2d(128, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )
        
        self.brightness_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Tanh()
        )

    def forward(self, z, y):
        # 1) shape synthesis
        y_emb = self.label_emb(y)                       # [B, latent_dim]
        z_feat = self.noise_proj(z).view(-1, 512, 4, 4)      # [B,512,4,4]
        y_feat = self.label_proj(y_emb).view(-1, 512, 4, 4)  # [B,512,4,4]
        x = torch.cat([z_feat, y_feat], dim=1)      # [B,1024,4,4]
        img = self.net(x)                             # [B,3,32,32] in [-1,+1]

        # 2) brightness factor from z
        # [B,1,1,1] in [-1,+1]
        b = self.brightness_head(z).view(-1, 1, 1, 1)
        # map to [0.5, 1.5] (or choose any sensible range)
        b = b * 0.5 + 1.0

        # 3) apply it
        return img * b                           # [B,3,32,32]
