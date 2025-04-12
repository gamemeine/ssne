import math
import random
import matplotlib.pyplot as plt
from collections import Counter

import numpy as np


def plot_class_distribution(dataset):
    class_counts = Counter()
    for _, label in dataset:
        class_counts[label] += 1

    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("Class Distribution")
    plt.xticks(classes)
    plt.show()

def preview(dataset, n_samples=9, classes=None, seed=None):
    if seed is not None:
        random.seed(seed)

    indices = random.sample(range(len(dataset)), n_samples)
    samples = [dataset[i] for i in indices]
    
    nrows = math.ceil(math.sqrt(n_samples))
    ncols = math.ceil(n_samples / nrows)
    
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axs = axs.flatten() if n_samples > 1 else [axs]
    
    for i, (img, label) in enumerate(samples):
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0, 1)
        axs[i].imshow(img_np)
        title = classes[label] if (classes is not None and isinstance(label, int)) else str(label)
        axs[i].set_title(title)
        axs[i].axis("off")
    
    for j in range(i + 1, len(axs)):
        axs[j].axis("off")
    
    plt.tight_layout()
    plt.show()