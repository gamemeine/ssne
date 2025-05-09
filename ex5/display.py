import math
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_images(images: list[torch.Tensor]):
    count = len(images)
    if count == 0:
        print("No images to display.")
        return

    cols = min(5, count)
    rows = math.ceil(count / cols)

    plt.figure(figsize=(cols * 2, rows * 2))
    for idx, img in enumerate(images):
        arr = img.detach().cpu().numpy()    # Convert to numpy array
        arr = np.transpose(arr, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(arr.squeeze())
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()