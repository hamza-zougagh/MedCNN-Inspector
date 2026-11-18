from .train import fit_linear


def fit_probe(train, val):
    return fit_linear(
        (train["intermediate"], train["labels"]), (val["intermediate"], val["labels"])
    )
