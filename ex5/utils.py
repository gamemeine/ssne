import torch


def denormalize_batch(x: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    mean = torch.tensor(mean, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(std,  device=x.device).view(1, -1, 1, 1)
    return x * std + mean
