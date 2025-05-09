import math
import matplotlib.pyplot as plt


def plot_images(images):
    count = len(images)
    if count == 0:
        print("No images to display.")
        return

    cols = min(5, count)
    rows = math.ceil(count / cols)

    plt.figure(figsize=(cols * 2, rows * 2))
    for idx in range(count):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(images[idx])
        ax.axis('off')
    plt.tight_layout()
    plt.show()