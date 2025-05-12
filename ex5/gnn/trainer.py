import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.normalization import denormalize_batch
from utils.display import plot_images


class Trainer:
    def __init__(self, n_classes, adversarial_criterion, latent_dim, device='cpu'):
        self.adversarial_criterion = adversarial_criterion
        self.latent_dim = latent_dim
        self.device = device

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

    def preview(self, mean=[0.5]*3, std=[0.5]*3):
        self.generator.eval()
        with torch.no_grad():
            fake_norm = self.generator(self.fixed_noise, self.fixed_labels).cpu()
            fake = denormalize_batch(fake_norm, mean=mean, std=std)
        plot_images(list(fake), ncols=9)

    def fit(self, dataloader: DataLoader, num_epochs: int = 100, mean=[0.5]*3, std=[0.5]*3):
        G_losses, D_losses = [], []
        for epoch in range(num_epochs):
            self.generator.train()

            D_fake_probs, D_real_probs = [], []
            epoch_D_loss, epoch_G_loss = 0.0, 0.0
            steps = 0

            for real_images, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                b_size = real_images.size(0)

                # ---------------------
                # 1) Update D network
                # ---------------------
                self.discriminator_optimizer.zero_grad()

                # Real batch
                rf_real = self.discriminator(real_images, labels)
                prob_real = torch.sigmoid(rf_real)
                D_real_probs.append(prob_real.mean().item())
                valid = torch.ones(b_size, device=self.device)
                loss_D_real = self.adversarial_criterion(rf_real, valid)

                # Fake batch
                noise = torch.randn(
                    b_size, self.latent_dim, device=self.device)
                fake_images = self.generator(noise, labels)
                rf_fake = self.discriminator(fake_images.detach(), labels)
                prob_fake = torch.sigmoid(rf_fake)
                D_fake_probs.append(prob_fake.mean().item())
                fake = torch.zeros(b_size, device=self.device)
                loss_D_fake = self.adversarial_criterion(rf_fake, fake)

                loss_D = loss_D_real + loss_D_fake
                loss_D.backward()
                self.discriminator_optimizer.step()

                # ---------------------
                # 2) Update G network
                # ---------------------
                self.generator_optimizer.zero_grad()
                rf_fake2 = self.discriminator(fake_images, labels)
                # we want D(G(z)) → 1
                valid = torch.ones(b_size, device=self.device)
                loss_G = self.adversarial_criterion(rf_fake2, valid)
                loss_G.backward()
                self.generator_optimizer.step()

                # record
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())
                epoch_D_loss += loss_D.item()
                epoch_G_loss += loss_G.item()
                steps += 1

            # Scheduler step
            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            # compute epoch‐level averages
            avg_D_real = np.mean(D_real_probs)
            avg_D_fake = np.mean(D_fake_probs)
            avg_loss_D = epoch_D_loss / steps
            avg_loss_G = epoch_G_loss / steps

            print((
                f"Epoch {epoch+1}: "
                f"D(real)={avg_D_real:.3f}  D(fake)={avg_D_fake:.3f}  "
                f"D_loss={avg_loss_D:.3f}  G_loss={avg_loss_G:.3f}"
            ))

            # Visualize every 10 epochs
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.preview(mean=mean, std=std)

        # Final visualization
        self.preview(mean=mean, std=std)
        return {'G_losses': G_losses, 'D_losses': D_losses}