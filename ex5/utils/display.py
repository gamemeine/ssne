import math
import PIL
import PIL.Image
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional


def plot_images(images: list[torch.Tensor | PIL.Image.Image], ncols: int = 5, title: Optional[str] = None):
    count = len(images)
    if count == 0:
        print("No images to display.")
        return

    cols = min(count, ncols)
    rows = math.ceil(count / cols)

    def subplot_tensor(ax: Axes, img: torch.Tensor):
        arr = img.detach().cpu().numpy()    # Convert to numpy array
        arr = np.transpose(arr, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
        ax.imshow(arr.squeeze())

    def subplot_pil(ax: Axes, img: PIL.Image.Image):
        ax.imshow(img)
        
    plt.figure(figsize=(cols * 2, rows * 2))

    if title:
        plt.suptitle(title, fontsize=14, y=1.0)


    for idx, img in enumerate(images):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.axis('off')

        if type(img) is torch.Tensor:
            subplot_tensor(ax, img)
        elif type(img) is PIL.Image.Image:
            subplot_pil(ax, img)
        else:
            raise Exception('Unknown type to display image: ' + type(img))
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.97 if title else 1.])
    plt.show()