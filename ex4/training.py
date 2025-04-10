import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def calculate_accuracy(correct_preds, total_preds, total_classes=50):
    class_accuracies = []
    for cls in range(total_classes):
        if total_preds[cls] > 0:
            accuracy = correct_preds[cls].item() / total_preds[cls].item()
            class_accuracies.append(accuracy)
        else:
            class_accuracies.append(0.0)

    average_class_accuracy = sum(class_accuracies) / total_classes
    return average_class_accuracy


class Trainer:
    def __init__(self, model, train_dl, val_dl, device='cpu', lr=1e-3, total_classes=50):
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.total_classes = total_classes

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_step(self, epoch, num_epochs):
        self.model.train()
        total_loss = 0.0
        for images, labels in tqdm(self.train_dl, desc=f"Epoch {epoch}/{num_epochs} (training)", leave=True):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)

        epoch_loss = total_loss / len(self.train_dl.dataset)
        return epoch_loss

    def evaluation_step(self, epoch, num_epochs):
        self.model.eval()
        total_loss = 0.0

        correct_preds = torch.zeros(self.total_classes, dtype=torch.int)
        total_preds = torch.zeros(self.total_classes, dtype=torch.int)

        with torch.no_grad():
            for images, labels in tqdm(self.val_dl, desc=f"Epoch {epoch}/{num_epochs} (validation)", leave=True):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, 1)

                for label, pred in zip(labels, preds):
                    total_preds[label] += 1
                    if pred == label:
                        correct_preds[label] += 1

        val_loss = total_loss / len(self.val_dl.dataset)
        avg_class_acc = calculate_accuracy(correct_preds, total_preds)

        return val_loss, avg_class_acc

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_step(epoch, num_epochs)
            val_loss, avg_class_acc = self.evaluation_step(epoch, num_epochs)

            print(f"Epoch {epoch}/{num_epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Avg Class Accuracy: {avg_class_acc:.4f}")
