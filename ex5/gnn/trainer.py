import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.normalization import denormalize_batch
from utils.display import plot_images


class Trainer:
    def __init__(self, n_classes, adversarial_criterion, classification_criterion, latent_dim, device='cpu'):
        self.adversarial_criterion = adversarial_criterion
        self.classification_criterion = classification_criterion
        self.latent_dim = latent_dim
        self.device = device

        # For visualization: one sample per class
        self.fixed_noise = torch.randn(n_classes, latent_dim, device=device)
        self.fixed_labels = torch.arange(n_classes, device=device)

    def set_discriminator(self, discriminator, discriminator_optimizer, discriminator_scheduler):
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_scheduler = discriminator_scheduler

    def set_generator(self, generator, generator_optimizer, generator_scheduler):
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.generator_scheduler = generator_scheduler

    def fit(self, dataloader: DataLoader, num_epochs: int = 100, mean=[0.5]*3, std=[0.5]*3):
        G_losses, D_losses = [], []
        for epoch in range(num_epochs):
            D_fake_acc, D_real_acc = [], []
            for real_images, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                b_size = real_images.size(0)

                # ---------------------
                # 1) Update D network
                # ---------------------
                self.discriminator_optimizer.zero_grad()
                # Real batch
                rf_real, cls_real = self.discriminator(real_images)
                valid = torch.ones(b_size, device=self.device)
                loss_D_real = self.adversarial_criterion(rf_real, valid)
                loss_cls_real = self.classification_criterion(cls_real, labels)
                D_real_acc.append(rf_real.mean().item())

                # Fake batch
                noise = torch.randn(b_size, self.latent_dim, device=self.device)
                fake_images = self.generator(noise, labels)
                rf_fake, cls_fake = self.discriminator(fake_images.detach())
                fake = torch.zeros(b_size, device=self.device)
                loss_D_fake = self.adversarial_criterion(rf_fake, fake)
                loss_cls_fake = self.classification_criterion(cls_fake, labels)
                D_fake_acc.append(rf_fake.mean().item())

                # Total D loss
                loss_D = loss_D_real + loss_D_fake + loss_cls_real + loss_cls_fake
                loss_D.backward()
                self.discriminator_optimizer.step()

                # ---------------------
                # 2) Update G network
                # ---------------------
                self.generator_optimizer.zero_grad()
                rf_fake2, cls_fake2 = self.discriminator(fake_images)
                valid = torch.ones(b_size, device=self.device)
                loss_G_adv = self.adversarial_criterion(rf_fake2, valid)
                loss_G_cls = self.classification_criterion(cls_fake2, labels)
                loss_G = loss_G_adv + loss_G_cls
                loss_G.backward()
                self.generator_optimizer.step()

                # Record losses
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())

            # Scheduler step
            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            print(f"D_fake_acc={np.mean(D_fake_acc):.3f}, D_real_acc={np.mean(D_real_acc):.3f}")

            # Visualize every 10 epochs
            if epoch % 10 == 0:
                with torch.no_grad():
                    fake_norm = self.generator(self.fixed_noise, self.fixed_labels).cpu()
                    fake = denormalize_batch(fake_norm, mean=mean, std=std).clamp(0, 1)
                plot_images(list(fake), ncols=9)

        # Final visualization
        with torch.no_grad():
            fake_norm = self.generator(self.fixed_noise, self.fixed_labels).cpu()
            fake = denormalize_batch(fake_norm, mean=mean, std=std).clamp(0, 1)
        plot_images(list(fake), ncols=9)

        return {'G_losses': G_losses, 'D_losses': D_losses}