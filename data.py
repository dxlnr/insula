"""Dataset"""
import torch


def def_lookup_table(words: list[str]) -> dict:
    """Returns a lookup table with all the chars in words plus special char
    for start and ending."""
    clist = ["."] + sorted(list(set("".join(words))))
    return {s: i for i, s in enumerate(clist)}


def get_dataset(words: list[str], lt: dict[str, int], block_size: int = 3):
    """Prepares a training dataset from list of words."""
    x, y = [], []
    for w in words:

        context = [0] * block_size
        for i, ch in enumerate(w + "."):
            x.append(context)
            y.append(lt[ch])
            context = context[1:] + [lt[ch]]

    return torch.Tensor(x), torch.Tensor(y)


