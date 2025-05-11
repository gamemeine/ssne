import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), 
            nn.LeakyReLU(0.2, True),
            
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.BatchNorm2d(128), 
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, 4, 2, 1), 
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, True),
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
        self.label_emb = nn.Embedding(n_classes, n_classes)
        # project combined (z + label) into 256×4×4
        self.fc = nn.Linear(latent_dim + n_classes, 256*4*4)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), 
            nn.BatchNorm2d(128), 
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  
            nn.BatchNorm2d(64),  
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 3, 4, 2, 1),    
            nn.Tanh()   # → [–1,+1]
        )

    def forward(self, z, y):
        # z: [B, latent_dim], y: [B] int labels
        y_vec = self.label_emb(y)                 # → [B, latent_dim]
        x = torch.cat([z, y_vec], dim=1)          # → [B, latent_dim*2]
        x = self.fc(x).view(-1, 256, 4, 4)         # → [B,256,4,4]
        return self.net(x)                        # → [B,3,32,32]
