import torch.nn as nn


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
        self.label_emb = nn.Embedding(n_classes, latent_dim)

        # 1) latent → 1024×4×4
        self.fc = nn.Linear(latent_dim, 1024*4*4)

        # 2) three ResBlock+upsample stacks to go 4→8→16→32
        self.up1 = nn.Sequential(
            ResBlock(1024),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024, 512, 1)   # project 1024→512
        )
        self.up2 = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 1)    # project 512→256
        )
        self.up3 = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 1)    # project 256→128
        )

        # 3) final BN+ReLU, conv→3, Tanh
        self.post = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z, y):
        y_emb = self.label_emb(y)
        x = z + y_emb
        x = self.fc(x).view(-1, 1024, 4, 4)
        x = self.up1(x)    # → [B,512, 8,  8]
        x = self.up2(x)    # → [B,256,16, 16]
        x = self.up3(x)    # → [B,128,32, 32]
        return self.post(x)


class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        # from RGB to 64
        self.from_rgb = nn.Conv2d(3, 64, 1)

        # 64 → ResBlock → down → project→ …
        self.stage1 = nn.Sequential(
            ResBlock(64),
            nn.AvgPool2d(2)    # 32→16
        )
        self.proj12 = nn.Conv2d(64, 128, 1)
        self.stage2 = nn.Sequential(
            ResBlock(128),
            nn.AvgPool2d(2)    # 16→8
        )
        self.proj23 = nn.Conv2d(128, 256, 1)
        self.stage3 = nn.Sequential(
            ResBlock(256),
            nn.AvgPool2d(2)    # 8→4
        )

        # widen to 512 at 4×4
        self.proj34 = nn.Conv2d(256, 512, 1)
        self.stage4 = nn.Sequential(
            ResBlock(512),
            nn.AvgPool2d(2)    # 4→2
        )
        self.proj45 = nn.Conv2d(512, 1024, 1)

        self.res_final = ResBlock(1024)   # no spatial change

        # activation + global sum-pool
        self.post_relu = nn.ReLU()

        # projection‐critic head
        self.adv_head = nn.Linear(1024, 1)
        self.label_emb = nn.Embedding(n_classes, 1024)

    def forward(self, x, y):
        x = self.from_rgb(x)        # [B,64,32,32]
        x = self.stage1(x)          # [B,64,16,16]
        x = self.proj12(x)
        x = self.stage2(x)          # [B,128, 8, 8]
        x = self.proj23(x)
        x = self.stage3(x)          # [B,256, 4, 4]
        x = self.proj34(x)          # [B,512, 4, 4]
        x = self.stage4(x)          # [B,512, 2, 2]
        x = self.proj45(x)          # [B,1024,2, 2]
        x = self.res_final(x)       # [B,512, 4, 4]
        x = self.post_relu(x)

        h = x.sum(dim=(2, 3))        # global sum-pool → [B,512]
        out = self.adv_head(h).view(-1)
        proj = (h * self.label_emb(y)).sum(1)
        return out + proj
