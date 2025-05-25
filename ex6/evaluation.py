import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def predict(model, val_dl, device='cpu'):
    model.eval()
    pred_labels = []
    true_labels = []
    with torch.no_grad():
        for seqs, lengths, labels in val_loader:
            seqs = seqs.to(device)
            lengths = lengths.to(device)
            logits = model(seqs, lengths)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            pred_labels.extend(preds)
            true_labels.extend(labels.numpy())

    return pred_labels, true_labels


def plot_true_false_positives(pred_labels, true_labels, label_values = None):
    label_values = label_values if label_values else list(set(true_labels))
    num_classes = len(label_values)

    tp = np.zeros(num_classes, dtype=int)
    fp = np.zeros(num_classes, dtype=int)
    for pred, true in zip(all_preds, all_labels):
        if pred == true:
            tp[pred] += 1
        else:
            fp[pred] += 1

    x = np.arange(num_classes)
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, tp, width, label='True Positives')
    ax.bar(x + width/2, fp, width, label='False Positives')
    ax.set_xticks(x)
    ax.set_xticklabels([label_values[i] for i in range(num_classes)])
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('True vs False Positives per Class on Validation Set')
    ax.legend()
    plt.show()


def plot_confusion_matrix(pred_labels, true_labels, label_values=None):
    label_values = label_values if label_values else list(set(true_labels))
    num_classes = len(label_values)

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[label_values[i] for i in range(num_classes)])
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title('Confusion Matrix on Validation Set')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()