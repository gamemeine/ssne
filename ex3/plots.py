from matplotlib import pyplot as plt
import numpy as np

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
    classes = [0, 1, 2]
    labels = ["cheap", "average", "expensive"]

    TP = []
    FP = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_test == c))
        fp = np.sum((y_pred == c) & (y_test != c))
        TP.append(tp)
        FP.append(fp)

    # print accuracy
    print(f"Average class accuracy: {np.mean([tp / (tp + fp) for tp, fp in zip(TP, FP)]):.2f}")
    for c, tp, fp in zip(classes, TP, FP):
        print(f"\"{labels[c]}\" accuracy: {tp / (tp + fp):.2f}")

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
