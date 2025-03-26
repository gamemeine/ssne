import torch
from torch.optim import lr_scheduler
from plots import plot_training

class Trainer:
    def __init__(self, model):
        self.model = model
  
    def to_device(self, device):
        self.model.to(device)
        self.device = device
    
    def compile(self, optimizer, criterion):
        self.set_optimizer(optimizer)
        self.set_criterion(criterion)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def train(self, dataloader):
        self.model.train()

        total_loss = 0
        total_batches = len(dataloader)

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)

            loss = self.criterion(pred, y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss/total_batches

    def evaluate(self, dataloader):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_batches = len(dataloader)

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                preds = self.model(X)
                loss = self.criterion(preds, y)
                total_loss += loss.item()

                predictions = (preds > 0.5).sum(dim=1)
                total_correct += (predictions == y).sum().item()
                total_samples += y.size(0)

        accuracy = total_correct / total_samples
        return total_loss/total_batches, accuracy

    def fit(self, train_dl, val_dl, epochs):
        train_results, val_results = [], []

        for t in range(epochs):
            avg_train_loss = self.train(train_dl)

            _, train_acc = self.evaluate(train_dl)
            avg_val_loss, val_acc = self.evaluate(val_dl)

            self.scheduler.step(avg_val_loss)
            current_lr = self.scheduler.get_last_lr()[0]

            train_results.append((avg_train_loss, train_acc))
            val_results.append((avg_val_loss, val_acc))

            print(f"Epoch {t+1:2}/{epochs} - Train Loss: {avg_train_loss:4.2f} - Val Loss: {avg_val_loss:4.2f} - Val Acc: {val_acc:2.4f} - LR: {current_lr}")

        plot_training(train_results, val_results)


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(X)
        return pred.cpu().numpy()