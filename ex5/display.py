import math
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_images(images: list[torch.Tensor], ncols: int = 5, title: str | None = None):
    count = len(images)
    if count == 0:
        print("No images to display.")
        return

    cols = min(count, ncols)
    rows = math.ceil(count / cols)

    plt.figure(figsize=(cols * 2, rows * 2))

    if title:
        plt.suptitle(title, fontsize=14, y=1.0)

    for idx, img in enumerate(images):
        arr = img.detach().cpu().numpy()    # Convert to numpy array
        arr = np.transpose(arr, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            arr_normalized = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr_normalized = np.zeros_like(arr) if arr_min < 0 or arr_min > 1 else arr

        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(arr_normalized.squeeze()) 
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97 if title else 1.])
    plt.show()