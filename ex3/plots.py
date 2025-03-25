from matplotlib import pyplot as plt
import numpy as np

from utils import price_to_class


def plot_training(train_results: tuple[float, float], val_results: tuple[float, float]):
    train_losses, train_accs = zip(*train_results)
    val_losses, val_accs = zip(*val_results)

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()

    axes[1].plot(train_accs, label='Train Accuracy')
    axes[1].plot(val_accs, label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_pred_acc(y_test: np.ndarray, y_pred: np.ndarray):
    true_classes = np.array([price_to_class(price) for price in y_test])
    pred_classes = np.array([price_to_class(price) for price in y_pred])

    classes = [0, 1, 2]
    labels = ["cheap", "average", "expensive"]

    TP = []
    FP = []
    for c in classes:
        tp = np.sum((pred_classes == c) & (true_classes == c))
        fp = np.sum((pred_classes == c) & (true_classes != c))
        TP.append(tp)
        FP.append(fp)

    bar_width = 0.35
    r1 = np.arange(len(classes))
    r2 = r1 + bar_width

    plt.figure(figsize=(8, 6))
    plt.bar(r1, TP, width=bar_width, color='green', edgecolor='black', label='True Positives')
    plt.bar(r2, FP, width=bar_width, color='red', edgecolor='black', label='False Positives')

    plt.xticks(r1 + bar_width/2, labels)
    plt.xlabel("Price Class")
    plt.ylabel("Count")
    plt.title("TP and FP for each Price Class")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def plot_regression(y_test: np.ndarray, y_pred: np.ndarray):
    mask = np.array([price_to_class(p) == price_to_class(a)for a, p in zip(y_test, y_pred)])


    correct_actual = y_test[mask]
    correct_pred = y_pred[mask]

    incorrect_actual = y_test[~mask]
    incorrect_pred = y_pred[~mask]

    plt.figure(figsize=(10, 5))
    plt.scatter(correct_actual, correct_pred, alpha=0.5, label="Correct Predictions", color='blue')
    plt.scatter(incorrect_actual, incorrect_pred, alpha=0.5, label="Incorrect Predictions", color='red')

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted values')

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val],color='green', linestyle='--', label='Ideal')
    plt.legend()
    plt.show()
