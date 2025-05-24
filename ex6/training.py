import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler = None, device = 'cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler if scheduler else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        self.device = device

        self.all_train_losses = []
        self.all_train_accuracies = []
        self.all_val_losses = []
        self.all_val_accuracies = []


    def train(self, train_dl, val_dl, epochs):
        for epoch in range(epochs):
            self.model.train()

            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_seqs, batch_lengths, batch_labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_seqs = batch_seqs.to(self.device)
                batch_lengths = batch_lengths.to(self.device)
                batch_labels = batch_labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(batch_seqs, batch_lengths)
                loss = self.criterion(logits, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * batch_seqs.size(0)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
            
            avg_loss = total_loss / total_samples
            train_acc = total_correct / total_samples

            val_loss, val_acc = self.evaluate(val_dl)
            self.scheduler.step(val_loss)

            self.all_train_losses.append(avg_loss)
            self.all_train_accuracies.append(train_acc)
            self.all_val_losses.append(val_loss)
            self.all_val_accuracies.append(val_acc)

            current_lr = self.scheduler.get_last_lr()
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr[0]:.6f}")


    def evaluate(self, val_loader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        with torch.no_grad():
            for batch_seqs, batch_lengths, batch_labels in val_loader:
                batch_seqs = batch_seqs.to(self.device)
                batch_lengths = batch_lengths.to(self.device)
                batch_labels = batch_labels.to(self.device)
                logits = self.model(batch_seqs, batch_lengths)
                loss = self.criterion(logits, batch_labels)
                preds = torch.argmax(logits, dim=1)
                total_correct += (preds == batch_labels).sum().item()
                total_samples += batch_labels.size(0)
                total_loss += loss.item() * batch_seqs.size(0)
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def plot_training_history(self):
        fig, ax = plt.subplots(1, 2, figsize=(16, 5))

        # Plot training and validation loss
        ax[0].plot(self.all_train_losses, label='Train Loss')
        ax[0].plot(self.all_val_losses, label='Validation Loss')
        ax[0].set_title('Loss over epochs')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()

        # Plot training and validation accuracy
        ax[1].plot(self.all_train_accuracies, label='Train Accuracy')
        ax[1].plot(self.all_val_accuracies, label='Validation Accuracy')
        ax[1].set_title('Accuracy over epochs')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")