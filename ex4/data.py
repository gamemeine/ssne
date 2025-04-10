import os
import shutil
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch import Generator


def train_val_split(base_dir, output_train_dir, output_val_dir, val_ratio=0.2, random_state=42):
    base_ds = ImageFolder(base_dir)
    total_samples = len(base_ds)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size

    train_ds, val_ds = random_split(base_ds, [train_size, val_size], generator=Generator().manual_seed(random_state))

    save_subset(train_ds, base_ds, output_train_dir)
    save_subset(val_ds, base_ds, output_val_dir)


def save_subset(subset, original_dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for idx in subset.indices:
        filepath, label = original_dataset.samples[idx]
        class_name = original_dataset.classes[label]
        dest_dir = os.path.join(output_dir, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(filepath, dest_dir)