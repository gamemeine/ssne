import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np


def plot_results_over_params(param_values: np.ndarray, results: np.ndarray, param_name: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    losses, accs = results[:, 0], results[:, 1]

    axes[0].plot(param_values, accs)
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy over ' + param_name)
    axes[0].tick_params(axis='y')

    axes[1].plot(param_values, losses)
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss over ' + param_name)
    axes[1].tick_params(axis='y')

    fig.tight_layout()
    plt.show()

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

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    classes = [0, 1, 2]
    labels = ["cheap", "average", "expensive"]

    confusion_matrix = np.zeros((3, 3))
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true, pred] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks(classes, labels)
    plt.yticks(classes, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center', color='black')
    plt.show()


def plot_pred_acc(y_test: np.ndarray, y_pred: np.ndarray):
    classes = [0, 1, 2]
    labels = ["cheap", "average", "expensive"]

    class_counts = np.bincount(y_test)

    TP = []
    FP = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_test == c))
        fp = np.sum((y_pred == c) & (y_test != c))
        TP.append(tp)
        FP.append(fp)

    # print accuracy
    class_accs = []
    for c in classes:
        class_records = y_test == c
        class_acc = np.mean(y_pred[class_records] == c)
        class_accs.append(class_acc)
        print(f"Class {c} accuracy: {class_acc:.2f}")
    
    print(f"Average class accuracy: {np.mean(class_accs):.2f}")

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


def plot_acc_over_thresholds(y_val: np.ndarray, y_pred: np.ndarray):
    thresholds = np.linspace(0, 1, 100)

    acc_class0 = []
    acc_class1 = []
    acc_class2 = []

    for t in thresholds:
        pred_class = ((y_pred[:, 0] >= t).astype(
            int) + (y_pred[:, 1] >= t).astype(int))

        mask0 = (y_val == 0)
        mask1 = (y_val == 1)
        mask2 = (y_val == 2)

        acc0 = np.mean(pred_class[mask0] == y_val[mask0]
                       ) if np.any(mask0) else np.nan
        acc1 = np.mean(pred_class[mask1] == y_val[mask1]
                       ) if np.any(mask1) else np.nan
        acc2 = np.mean(pred_class[mask2] == y_val[mask2]
                       ) if np.any(mask2) else np.nan

        acc_class0.append(acc0)
        acc_class1.append(acc1)
        acc_class2.append(acc2)

    avg_acc = np.nanmean(np.array([acc_class0, acc_class1, acc_class2]), axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, acc_class0, label='Class 0 Accuracy')
    plt.plot(thresholds, acc_class1, label='Class 1 Accuracy')
    plt.plot(thresholds, acc_class2, label='Class 2 Accuracy')
    plt.plot(thresholds, avg_acc, label='Average Accuracy',linestyle='--', color='black')
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy over Thresholds")
    plt.legend()
    plt.grid(True)
    plt.show()
