import torch


def price_to_class(price: float) -> int:
    if price <= 100_000:
        return 0  # cheap
    elif price <= 350_000:
        return 1  # average
    else:
        return 2  # expensive
    
def pred_to_class(pred, threshold=0.5):
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    
    probs = torch.sigmoid(pred)
    return (probs > threshold).sum(dim=1)