import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(h + x)


class Generator(nn.Module):
    def __init__(self, n_classes, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(n_classes, latent_dim)

        # 1) FC → 256×4×4
        self.fc = nn.Linear(latent_dim, 256*4*4)

        # 2) three (ResBlock + Upsample) stacks to go 4→8→16→32
        self.res_up1 = nn.Sequential(
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.res_up2 = nn.Sequential(
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.res_up3 = nn.Sequential(
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

        # 3) final BN + ReLU, then 3×3 conv → Tanh
        self.post = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, y):
        """
        z: [B, latent_dim], y: [B]
        """
        y_emb = self.label_emb(y)       # [B, latent_dim]
        x = z + y_emb                   # simple additive conditioning
        x = self.fc(x).view(-1, 256, 4, 4)
        x = self.res_up1(x)             # → 256×8×8
        x = self.res_up2(x)             # → 256×16×16
        x = self.res_up3(x)             # → 256×32×32
        return self.post(x)             # → 3×32×32 in [-1,+1]


class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # 1) Expand 3→128, then two ResBlock+pool stages
        self.conv_in = nn.Conv2d(3, 256, 1)
        self.res_dn1 = nn.Sequential(
            ResBlock(256),
            nn.AvgPool2d(2)    # 32→16
        )
        self.res_dn2 = nn.Sequential(
            ResBlock(256),
            nn.AvgPool2d(2)    # 16→8
        )

        # 2) two more ResBlocks at 128×8×8
        self.res3 = ResBlock(256)
        self.res4 = ResBlock(256)

        # 3) Global sum‐pool → projection head
        self.post_relu = nn.ReLU()
        feat_dim = 256
        self.adv_head = nn.Linear(feat_dim, 1)
        self.label_emb = nn.Embedding(n_classes, feat_dim)

    def forward(self, x, y):
        x = self.conv_in(x)   # [B,128,32,32]
        x = self.res_dn1(x)   # [B,128,16,16]
        x = self.res_dn2(x)   # [B,128, 8, 8]
        x = self.res3(x)      # [B,128, 8, 8]
        x = self.res4(x)      # [B,128, 8, 8]
        x = self.post_relu(x)
        h = x.sum((2, 3))      # [B,128]
        real_logit = self.adv_head(h).view(-1)
        v_y = self.label_emb(y)
        proj = (h * v_y).sum(1)
        return real_logit + proj
