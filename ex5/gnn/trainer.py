import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from utils import denormalize_batch
from display import plot_images


class Trainer:
    def __init__(self, criterion, latent_dim, device = 'cpu'):
        self.criterion = criterion
        self.latent_dim = latent_dim
        self.device = device

        self.fixed_noise = torch.randn(10, latent_dim, device=device)

    def set_discriminator(self, discriminator, discriminator_optimizer, discriminator_scheduler):
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_scheduler = discriminator_scheduler

    def set_generator(self, generator, generator_optimizer, generator_scheduler):
        self.generator = generator
        self.generator_optimizer = generator_optimizer
        self.generator_scheduler = generator_scheduler

    def fit(self, dataloader: DataLoader, num_epochs: int = 100):
        G_losses = []
        D_losses = []
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            discriminator_fake_acc = []
            discriminator_real_acc = []
            for i, data in enumerate(dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                self.discriminator_optimizer.zero_grad()
                # Format batch
                real_images = data[0].to(self.device)
                b_size = real_images.size(0)
                # Setting labels for real images
                label = torch.ones((b_size,), dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.discriminator(real_images).view(-1)
                # Calculate loss on all-real batch
                error_discriminator_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                discriminator_real_acc.append(output.mean().item())

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.latent_dim, device=self.device)
                # Generate fake image batch with Generator
                fake_images = self.generator(noise)
                label_fake = torch.zeros((b_size,), dtype=torch.float, device=self.device)
                # Classify all fake batch with Discriminator
                output = self.discriminator(fake_images.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                error_discriminator_fake = self.criterion(output, label_fake)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                discriminator_fake_acc.append(output.mean().item())
                # Compute error of D as sum over the fake and the real batches
                error_discriminator = error_discriminator_real + error_discriminator_fake
                error_discriminator.backward()
                # Update D
                self.discriminator_optimizer.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.generator_optimizer.zero_grad()
                # fake labels are real for generator cost
                label = torch.ones((b_size,), dtype=torch.float, device=self.device)
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.discriminator(fake_images).view(-1)
                # Calculate G's loss based on this output
                error_generator = self.criterion(output, label)
                # Calculate gradients for G
                error_generator.backward()
                # Update G
                self.generator_optimizer.step()

                # Output training stats
                # Save Losses for plotting later
                G_losses.append(error_generator.item())
                D_losses.append(error_discriminator.item())

            print(f"Epoch: {epoch}, discrimiantor fake error: {np.mean(discriminator_fake_acc):.3}, discriminator real acc: {np.mean(discriminator_real_acc):.3}")
            self.generator_scheduler.step()
            self.discriminator_scheduler.step()

            if epoch % 10 == 0:
                with torch.no_grad():
                    fake_norm = self.generator(self.fixed_noise).detach().cpu()
                    fake = denormalize_batch(fake_norm, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).clamp(0.0, 1.0)

                plot_images(list(fake))
