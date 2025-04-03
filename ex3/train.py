import numpy as np
import pandas as pd
import torch
from torch.optim import lr_scheduler
from utils import calc_accuracy, pred_to_class
from plots import plot_training

class Trainer:
    def __init__(self, model):
        self.model = model
        self.verbose = True
  
    def to_device(self, device):
        self.model.to(device)
        self.device = device
    
    def compile(self, optimizer, criterion, scheduler=None):
        self.set_optimizer(optimizer, scheduler)
        self.set_criterion(criterion)

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        if lr_scheduler is not None:
            self.scheduler = scheduler
        else:
            self.scheduler = scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def set_verbose(self, verbose):
        self.verbose = verbose

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
        total_samples = 0
        total_batches = len(dataloader)

        y_pred = np.array([])
        y_true = np.array([])

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                preds = self.model(X)
                loss = self.criterion(preds, y)
                total_loss += loss.item()

                predictions = (preds > 0.5).sum(dim=1)
                y_pred = np.concatenate((y_pred, predictions.cpu().numpy()))
                y_true = np.concatenate((y_true, y.cpu().numpy()))
                
                total_samples += y.size(0)

        accuracy = calc_accuracy(y_pred, y_true)
        return total_loss/total_batches, accuracy

    def fit(self, train_dl, val_dl, epochs, print_progress=True):
        train_results, val_results = [], []

        for t in range(epochs):
            avg_train_loss = self.train(train_dl)

            _, train_acc = self.evaluate(train_dl)
            avg_val_loss, val_acc = self.evaluate(val_dl)

            self.scheduler.step(avg_val_loss)
            current_lr = self.scheduler.get_last_lr()[0]

            train_results.append((avg_train_loss, train_acc))
            val_results.append((avg_val_loss, val_acc))

            if self.verbose:
                print(f"Epoch {t+1:2}/{epochs} - Train Loss: {avg_train_loss:4.2f} - Val Loss: {avg_val_loss:4.2f} - Val Acc: {val_acc:2.4f} - LR: {current_lr}")

        if self.verbose:
            plot_training(train_results, val_results)

        return val_results[-1]


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        pred = model(X)
        return pred.cpu().numpy()

def save_results(model, X, thresholds = (0.5, 0.5) , device = "cpu", path = 'pred.csv'):
    predictions = predict(model, X, device)
    classes = pred_to_class(predictions, thresholds)

    results = pd.DataFrame(classes)
    results.to_csv(path, index=False, header=False)

    return results