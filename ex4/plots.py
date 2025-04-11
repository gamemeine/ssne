import matplotlib.pyplot as plt
from collections import Counter


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
