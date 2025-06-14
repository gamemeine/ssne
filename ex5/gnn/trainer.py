import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.normalization import denormalize_batch
from utils.display import plot_images
from .loss import HingeLoss
from .models import ConditionalVariationalAutoencoder, BigConditionalVariationalAutoencoder
import torch.nn.functional as F


class Trainer:
    def __init__(self, n_classes, adversarial_criterion, latent_dim, device='cpu'):
        self.adversarial_criterion = adversarial_criterion
        self.latent_dim = latent_dim
        self.device = device

        self.fixed_noise = torch.randn(n_classes, latent_dim, device=device)
        self.fixed_labels = torch.arange(n_classes, device=device)

        self.hinge_loss = HingeLoss()
        self.model_id = time.time()

    def set_discriminator(self, discriminator, discriminator_optimizer, discriminator_scheduler):
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_scheduler = discriminator_scheduler

    def set_generator(self, generator, generator_optimizer, generator_scheduler):
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.generator_scheduler = generator_scheduler

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator.state_dict(), f'{path}/generator.pth')
        torch.save(self.discriminator.state_dict(), f'{path}/discriminator.pth')

    def preview(self, mean=[0.5]*3, std=[0.5]*3):
        self.generator.eval()
        with torch.no_grad():
            fake_norm = self.generator(
                self.fixed_noise, self.fixed_labels).cpu()
            fake = denormalize_batch(fake_norm, mean=mean, std=std)
        plot_images(list(fake), ncols=9)

    def fit(self, dataloader: DataLoader, num_epochs: int = 100, mean=[0.5]*3, std=[0.5]*3):
        G_losses, D_losses = [], []

        for epoch in range(num_epochs):
            self.generator.train()
            self.discriminator.train()

            D_real_probs, D_fake_probs = [], []
            epoch_D_loss, epoch_G_loss = 0.0, 0.0
            steps = 0

            for real_images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)
                b_size = real_images.size(0)

                # ---------------------
                # 1) Update Discriminator
                # ---------------------
                self.discriminator_optimizer.zero_grad()

                # Real images → patch logits [B,1,H,W]
                p_real = self.discriminator(real_images, labels)
                D_real_probs.append(torch.sigmoid(p_real).mean().item())

                # valid_map = torch.ones_like(p_real)
                # loss_D_real = self.adversarial_criterion(p_real, valid_map).mean()

                # Fake images → patch logits
                noise = torch.randn(b_size, self.latent_dim, device=self.device)
                fake_images = self.generator(noise, labels)

                p_fake = self.discriminator(fake_images.detach(), labels)
                D_fake_probs.append(torch.sigmoid(p_fake).mean().item())

                # fake_map = torch.zeros_like(p_fake)
                # loss_D_fake = self.adversarial_criterion(p_fake, fake_map).mean()

                # Total discriminator loss
                # loss_D = loss_D_real + loss_D_fake
                loss_D = self.hinge_loss.d_loss(p_real.view(-1), p_fake.view(-1))
                loss_D.backward()
                self.discriminator_optimizer.step()

                # ---------------------
                # 2) Update Generator
                # ---------------------
                self.generator_optimizer.zero_grad()

                # Re-run fake through D to get fresh gradients
                p_fake2 = self.discriminator(fake_images, labels)
                # valid_map2 = torch.ones_like(p_fake2)
                # loss_G = self.adversarial_criterion(p_fake2, valid_map2).mean()
                loss_G = self.hinge_loss.g_loss(p_fake2.view(-1))

                loss_G.backward()
                self.generator_optimizer.step()

                # record losses
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())
                epoch_D_loss += loss_D.item()
                epoch_G_loss += loss_G.item()
                steps += 1

            # step schedulers
            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            # compute epoch‐level averages
            avg_D_real = np.mean(D_real_probs)
            avg_D_fake = np.mean(D_fake_probs)
            avg_loss_D = epoch_D_loss / steps
            avg_loss_G = epoch_G_loss / steps

            print((
                f"Epoch {epoch+1}: "
                f"D(real)={avg_D_real:.3f}  "
                f"D(fake)={avg_D_fake:.3f}  "
                f"D_loss={avg_loss_D:.3f}  "
                f"G_loss={avg_loss_G:.3f}"
            ))

            # preview generated samples
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                self.preview(mean=mean, std=std)

            # save model
            self.save(f'./weights/{self.model_id}')

        # final preview
        self.preview(mean=mean, std=std)
        return {'G_losses': G_losses, 'D_losses': D_losses}


class cVAETrainer:
    def __init__(self, cvae_model: ConditionalVariationalAutoencoder, optimizer, num_classes: int, scheduler=None, latent_dim: int = 20, device: str = 'cpu'):
        self.model = cvae_model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device

        self.num_viz_samples = 32
        self.fixed_noise_for_generation = torch.randn(self.num_viz_samples, latent_dim, device=device)

        self.fixed_labels_for_generation = torch.arange(self.num_classes, device=device).repeat_interleave(self.num_viz_samples // self.num_classes +1)
        self.fixed_labels_for_generation = self.fixed_labels_for_generation[:self.num_viz_samples]


    def _calculate_loss(self, x_reconstructed, x_original, mu, logvar):
        recon_loss = F.mse_loss(x_reconstructed, x_original, reduction='sum')

        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + kld_loss
        return total_loss, recon_loss, kld_loss

    def fit(self, dataloader: DataLoader, num_epochs: int = 50, img_channels: int = 3):
        train_losses, recon_losses, kld_losses = [], [], []
        
        fixed_real_batch, fixed_labels = next(iter(dataloader))
        num_recon_viz = 16
        fixed_real_batch = fixed_real_batch[:num_recon_viz].to(self.device)
        fixed_labels = fixed_labels[:num_recon_viz].to(self.device)

        for epoch in range(num_epochs):
            epoch_total_loss, epoch_recon_loss, epoch_kld_loss = 0.0, 0.0, 0.0
            num_batches = 0

            self.model.train()
            
            for real_images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)

                b_size = real_images.size(0)

                self.optimizer.zero_grad()

                x_reconstructed, mu, logvar = self.model(real_images, labels)

                loss, recon_loss_val, kld_loss_val = self._calculate_loss(
                    x_reconstructed, real_images, mu, logvar
                )
                
                loss = loss / b_size
                recon_loss_val = recon_loss_val / b_size
                kld_loss_val = kld_loss_val / b_size

                loss.backward()
                self.optimizer.step()

                epoch_total_loss += loss.item()
                epoch_recon_loss += recon_loss_val.item()
                epoch_kld_loss += kld_loss_val.item()
                num_batches += 1

            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kld_loss = epoch_kld_loss / num_batches

            train_losses.append(avg_total_loss)
            recon_losses.append(avg_recon_loss)
            kld_losses.append(avg_kld_loss)

            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Total Loss: {avg_total_loss:.4f}, "
                f"Recon Loss: {avg_recon_loss:.4f}, "
                f"KLD Loss: {avg_kld_loss:.4f}"
            )

            if epoch % 10 == 0 or epoch == num_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    reconstructed_fixed, _, _ = self.model(fixed_real_batch, fixed_labels)
                    
                    comparison_images = []
                    for i in range(fixed_real_batch.size(0)):
                        comparison_images.append(fixed_real_batch[i])
                        comparison_images.append(reconstructed_fixed[i])
                    
                    plot_images(
                        [denormalize_batch(img.unsqueeze(0), mean=[0.5]*img_channels, std=[0.5]*img_channels).squeeze(0) for img in comparison_images],
                        ncols=8,
                        title=f"Epoch {epoch+1}: Original & Reconstructed"
                    )

                    generated_samples = self.model.generate(self.fixed_noise_for_generation, self.fixed_labels_for_generation)
                    plot_images(
                        [denormalize_batch(img.unsqueeze(0), mean=[0.5]*img_channels, std=[0.5]*img_channels).squeeze(0) for img in generated_samples],
                        ncols=8,
                        title=f"Epoch {epoch+1}: Generated Samples from Scratch (Labels: {self.fixed_labels_for_generation.cpu().numpy()[:8]}...)"
                    )
        
        return {
            'total_losses': train_losses,
            'reconstruction_losses': recon_losses,
            'kld_losses': kld_losses
        }


class BigcVAETrainer:
    def __init__(self, cvae_model: BigConditionalVariationalAutoencoder, optimizer, num_classes: int, scheduler=None, latent_dim: int = 20, device: str = 'cpu'):
        self.model = cvae_model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device

        self.num_viz_samples = 32
        self.fixed_noise_for_generation = torch.randn(self.num_viz_samples, latent_dim, device=device)

        self.fixed_labels_for_generation = torch.arange(self.num_classes, device=device).repeat_interleave(self.num_viz_samples // self.num_classes +1)
        self.fixed_labels_for_generation = self.fixed_labels_for_generation[:self.num_viz_samples]


    def _calculate_loss(self, x_reconstructed, x_original, mu, logvar):
        # recon_loss = F.mse_loss(x_reconstructed, x_original, reduction='sum')   # L2 loss
        recon_loss = F.l1_loss(x_reconstructed, x_original, reduction='sum')    # L1 loss

        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + kld_loss
        return total_loss, recon_loss, kld_loss

    def fit(self, dataloader: DataLoader, num_epochs: int = 50, img_channels: int = 3):
        train_losses, recon_losses, kld_losses = [], [], []
        
        fixed_real_batch, fixed_labels = next(iter(dataloader))
        num_recon_viz = 16
        fixed_real_batch = fixed_real_batch[:num_recon_viz].to(self.device)
        fixed_labels = fixed_labels[:num_recon_viz].to(self.device)

        for epoch in range(num_epochs):
            epoch_total_loss, epoch_recon_loss, epoch_kld_loss = 0.0, 0.0, 0.0
            num_batches = 0

            self.model.train()
            
            for real_images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                real_images = real_images.to(self.device)
                labels = labels.to(self.device)

                b_size = real_images.size(0)

                self.optimizer.zero_grad()

                x_reconstructed, mu, logvar = self.model(real_images, labels)

                loss, recon_loss_val, kld_loss_val = self._calculate_loss(
                    x_reconstructed, real_images, mu, logvar
                )
                
                loss = loss / b_size
                recon_loss_val = recon_loss_val / b_size
                kld_loss_val = kld_loss_val / b_size

                loss.backward()
                self.optimizer.step()

                epoch_total_loss += loss.item()
                epoch_recon_loss += recon_loss_val.item()
                epoch_kld_loss += kld_loss_val.item()
                num_batches += 1

            avg_total_loss = epoch_total_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_kld_loss = epoch_kld_loss / num_batches

            train_losses.append(avg_total_loss)
            recon_losses.append(avg_recon_loss)
            kld_losses.append(avg_kld_loss)

            if self.scheduler:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Total Loss: {avg_total_loss:.4f}, "
                f"Recon Loss: {avg_recon_loss:.4f}, "
                f"KLD Loss: {avg_kld_loss:.4f}"
            )

            if epoch % 20 == 0 or epoch == num_epochs - 1:
                self.model.eval()
                with torch.no_grad():
                    reconstructed_fixed, _, _ = self.model(fixed_real_batch, fixed_labels)
                    
                    comparison_images = []
                    for i in range(fixed_real_batch.size(0)):
                        comparison_images.append(fixed_real_batch[i])
                        comparison_images.append(reconstructed_fixed[i])
                    
                    plot_images(
                        [denormalize_batch(img.unsqueeze(0), mean=[0.5]*img_channels, std=[0.5]*img_channels).squeeze(0) for img in comparison_images],
                        ncols=8,
                        title=f"Epoch {epoch+1}: Original & Reconstructed"
                    )

                    generated_samples = self.model.generate(self.fixed_noise_for_generation, self.fixed_labels_for_generation)
                    plot_images(
                        [denormalize_batch(img.unsqueeze(0), mean=[0.5]*img_channels, std=[0.5]*img_channels).squeeze(0) for img in generated_samples],
                        ncols=8,
                        title=f"Epoch {epoch+1}: Generated Samples from Scratch (Labels: {self.fixed_labels_for_generation.cpu().numpy()[:8]}...)"
                    )
        
        return {
            'total_losses': train_losses,
            'reconstruction_losses': recon_losses,
            'kld_losses': kld_losses
        }
