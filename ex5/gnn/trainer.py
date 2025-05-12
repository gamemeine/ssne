import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.normalization import denormalize_batch
from utils.display import plot_images
from .models import VariationalAutoencoder
import torch.nn.functional as F


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


class VAETrainer:
    def __init__(self, vae_model: VariationalAutoencoder, optimizer, scheduler=None, latent_dim: int = 20, device: str = 'cpu'):
        self.model = vae_model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.latent_dim = latent_dim
        self.device = device

        self.fixed_noise_for_generation = torch.randn(32, latent_dim, device=device) 

    def _calculate_loss(self, x_reconstructed, x_original, mu, logvar):
        recon_loss = F.mse_loss(x_reconstructed, x_original, reduction='sum')

        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + kld_loss
        return total_loss, recon_loss, kld_loss

    def fit(self, dataloader: DataLoader, num_epochs: int = 50, img_channels: int = 3):
        train_losses, recon_losses, kld_losses = [], [], []
        
        fixed_real_batch, _ = next(iter(dataloader))
        fixed_real_batch = fixed_real_batch[:16].to(self.device)

        for epoch in range(num_epochs):
            epoch_total_loss, epoch_recon_loss, epoch_kld_loss = 0.0, 0.0, 0.0
            num_batches = 0

            self.model.train()
            
            for real_images, _ in tqdm(dataloader, desc=f"Epoch {epoch}"):
                real_images = real_images.to(self.device)
                b_size = real_images.size(0)

                self.optimizer.zero_grad()

                x_reconstructed, mu, logvar = self.model(real_images)

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

            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    reconstructed_fixed, _, _ = self.model(fixed_real_batch)
                    
                    comparison_images = []
                    for i in range(fixed_real_batch.size(0)):
                        comparison_images.append(fixed_real_batch[i])
                        comparison_images.append(reconstructed_fixed[i])
                    
                    plot_images(
                        [denormalize_batch(img.unsqueeze(0), mean=[0.0]*img_channels, std=[1.0]*img_channels).squeeze(0) for img in comparison_images],
                        ncols=8,
                        title=f"Epoch {epoch+1}: Original & Reconstructed"
                    )

                    generated_samples = self.model.decode(self.fixed_noise_for_generation)
                    plot_images(
                        [denormalize_batch(img.unsqueeze(0), mean=[0.0]*img_channels, std=[1.0]*img_channels).squeeze(0) for img in generated_samples],
                        ncols=8,
                        title=f"Epoch {epoch+1}: Generated Samples from Scratch"
                    )
        
        return {
            'total_losses': train_losses,
            'reconstruction_losses': recon_losses,
            'kld_losses': kld_losses
        }