import matplotlib.pyplot as plt
import numpy as np
import torch 


def show_image(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if np.issubdtype(image.dtype, np.floating):
        image = (image + 1) / 2.0

    image = np.transpose(image, (1, 2, 0))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
