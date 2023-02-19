"""Dataset"""
import torch


def create_dataset(words, lt):
    """Prepares a training dataset from list of words."""
    xs, ys = list(), list()

    for w in words:
        aw = ["."] + list(w) + ["."]
        for c1, c2 in zip(aw, aw[1:]):
            xs.append(lt[c1])
            ys.append(lt[c2])

    xs, ys = torch.tensor(xs), torch.tensor(ys)

    return (torch.nn.functional.one_hot(xs, len(lt)).float(), ys)


def def_lookup_table(words: list[str]) -> dict:
    """Returns a lookup table with all the chars in words plus special char
    for start and ending."""
    clist = ["."] + sorted(list(set("".join(words))))
    return {s: i for i, s in enumerate(clist)}
