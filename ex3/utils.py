def price_to_class(price: float) -> int:
    if price <= 100_000:
        return 0  # cheap
    elif price <= 350_000:
        return 1  # average
    else:
        return 2  # expensive