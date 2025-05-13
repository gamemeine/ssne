import torch.nn as nn
import torch.nn.functional as F
import torch


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

        self.fc = nn.Linear(latent_dim, 1024*4*4)

        self.up1 = nn.Sequential(
            ResBlock(1024),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(1024, 512, 1)
        )
        self.up2 = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(512, 256, 1)
        )
        self.up3 = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 1)
        )

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
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.post(x)


class Discriminator(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.from_rgb = nn.Conv2d(3, 64, 1)

        self.stage1 = nn.Sequential(
            ResBlock(64),
            nn.AvgPool2d(2)
        )
        self.proj12 = nn.Conv2d(64, 128, 1)
        self.stage2 = nn.Sequential(
            ResBlock(128),
            nn.AvgPool2d(2)
        )
        self.proj23 = nn.Conv2d(128, 256, 1)
        self.stage3 = nn.Sequential(
            ResBlock(256),
            nn.AvgPool2d(2)
        )

        self.proj34 = nn.Conv2d(256, 512, 1)
        self.stage4 = nn.Sequential(
            ResBlock(512),
            nn.AvgPool2d(2)
        )
        self.proj45 = nn.Conv2d(512, 1024, 1)

        self.res_final = ResBlock(1024)

        self.post_relu = nn.ReLU()

        self.adv_head = nn.Linear(1024, 1)
        self.label_emb = nn.Embedding(n_classes, 1024)

    def forward(self, x, y):
        x = self.from_rgb(x)
        x = self.stage1(x)
        x = self.proj12(x)
        x = self.stage2(x)
        x = self.proj23(x)
        x = self.stage3(x)
        x = self.proj34(x)
        x = self.stage4(x)
        x = self.proj45(x)
        x = self.res_final(x)
        x = self.post_relu(x)

        h = x.sum(dim=(2, 3))
        out = self.adv_head(h).view(-1)
        proj = (h * self.label_emb(y)).sum(1)
        return out + proj


class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_channels, num_classes, latent_dim, img_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
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
        encoder_output_flat_size = 128 * (self.img_size // 8)**2
        combined_encoder_input_size = encoder_output_flat_size + self.num_classes

        self.fc_mu = nn.Linear(combined_encoder_input_size, latent_dim)
        self.fc_logvar = nn.Linear(combined_encoder_input_size, latent_dim)

        # === DECODER ===
        combined_decoder_input_size = latent_dim + self.num_classes
        self.decoder_fc_output_size = 128 * (self.img_size // 8)**2
        self.decoder_fc = nn.Linear(combined_decoder_input_size, self.decoder_fc_output_size)

        # Convolutional layers
        self.decoder_conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> [B, 64, 8, 8]
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> [B, 32, 16, 16]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, input_channels, kernel_size=4, stride=2, padding=1), # -> [B, input, 32, 32]
            nn.Tanh()
        )

    def encode(self, x, y_one_hot):
        h_conv = self.encoder_conv(x)
        h_combined = torch.cat((h_conv, y_one_hot), dim=1)
        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std

    def decode(self, z, y_one_hot):
        zy_combined = torch.cat((z, y_one_hot), dim=1)
        h = self.decoder_fc(zy_combined)
        h = h.view(-1, 128, self.img_size // 8, self.img_size // 8)
        return self.decoder_conv_transpose(h)

    def forward(self, x, y):
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float().to(x.device)

        mu, logvar = self.encode(x, y_one_hot)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z, y_one_hot)
        return x_reconstructed, mu, logvar

    def generate(self, z, y_int_labels):
        y_one_hot = F.one_hot(y_int_labels, num_classes=self.num_classes).float().to(z.device)
        return self.decode(z, y_one_hot)


class BigConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_channels, num_classes, latent_dim, img_size=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = img_size

        self.final_conv_output_spatial_dim = self.img_size // 16

        # === ENCODER ===
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # -> x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten()
        )

        # Layers fully connected 
        encoder_output_flat_size = 512 * (self.final_conv_output_spatial_dim)**2
        combined_encoder_input_size = encoder_output_flat_size + self.num_classes

        self.fc_mu = nn.Linear(combined_encoder_input_size, latent_dim)
        self.fc_logvar = nn.Linear(combined_encoder_input_size, latent_dim)

        # === DECODER ===
        combined_decoder_input_size = latent_dim + self.num_classes
        self.decoder_fc_output_size = 512 * (self.final_conv_output_spatial_dim)**2
        self.decoder_fc = nn.Linear(combined_decoder_input_size, self.decoder_fc_output_size)

        self.decoder_bn_fc = nn.BatchNorm1d(self.decoder_fc_output_size)
        self.decoder_relu_fc = nn.LeakyReLU(0.2, inplace=True)

        # Convolutional layers
        self.decoder_conv_transpose = nn.Sequential(

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def encode(self, x, y_one_hot):
        h_conv = self.encoder_conv(x)
        h_combined = torch.cat((h_conv, y_one_hot), dim=1)
        mu = self.fc_mu(h_combined)
        logvar = self.fc_logvar(h_combined)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return mu + eps * std

    def decode(self, z, y_one_hot):
        zy_combined = torch.cat((z, y_one_hot), dim=1)
        h = self.decoder_fc(zy_combined)
        h = self.decoder_bn_fc(h)
        h = self.decoder_relu_fc(h)
        h = h.view(-1, 512, self.final_conv_output_spatial_dim, self.final_conv_output_spatial_dim)
        return self.decoder_conv_transpose(h)

    def forward(self, x, y):
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float().to(x.device)

        mu, logvar = self.encode(x, y_one_hot)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z, y_one_hot)
        return x_reconstructed, mu, logvar

    def generate(self, z, y_int_labels):
        y_one_hot = F.one_hot(y_int_labels, num_classes=self.num_classes).float().to(z.device)
        return self.decode(z, y_one_hot)
