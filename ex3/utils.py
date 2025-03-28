import numpy as np
import torch


def price_to_class(price: float) -> int:
    if price <= 100_000:
        return 0  # cheap
    elif price <= 350_000:
        return 1  # average
    else:
        return 2  # expensive
    
def pred_to_class(pred, threshold=(0.5, 0.5)):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    
    probs = torch.sigmoid(pred)

    target_1 = (probs[:, 0] >= threshold[0]).int()
    target_2 = (probs[:, 1] >= threshold[1]).int()

    return target_1 + target_2


def calc_accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    accuracies = []
    for i in range(3):
        class_correct = (y_pred == y_true)[y_true == i].sum()
        accuracies.append(class_correct/(y_true == i).sum())
    return (np.mean(accuracies))
