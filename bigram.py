"""Various Language Models."""
from typing import Dict, List

import torch

from data import create_dataset


def def_lookup_table(words: List[str]):
    """Returns a lookup table with all the chars in words plus special char
    for start and ending."""
    clist = ["."] + sorted(list(set("".join(words))))
    return {s: i for i, s in enumerate(clist)}


def invert_lookup(lt):
    """Flip keys & values."""
    return {i: s for s, i in lt.items()}


def bigram(words: List[str], lt: Dict[str, int]) -> torch.Tensor:
    """Returns the Bigram of a list of str.

    :param words: Input data which is a list of strs.
    :param lt: Lookup table defining the elements used in dataset.
    """
    bigram = torch.zeros([len(lt), len(lt)], dtype=torch.int32)

    for w in words:
        aw = ["."] + list(w) + ["."]
        for c1, c2 in zip(aw, aw[1:]):
            bigram[lt[c1]][lt[c2]] += 1

    return bigram


def norm_bigram(bigram):
    """Normalize the tensor along each row."""
    bigram = smoothing(bigram).float()
    return bigram / bigram.sum(1, keepdim=True)


def smoothing(bigram, s_v: int = 1):
    """Smooth out the model in order to avoid nll going to inf."""
    return bigram + s_v


def sample_from_bigram(n_bigram, lt, seed: int = 2329723):
    """Use the bigram params to extract new sample.

    :param bigram: Normalized 2D bigram matrix storing the params as probabilities.
    :param lt: Lookup table.
    :param seed: Random seed for generator function.
    """
    g = torch.Generator().manual_seed(seed)
    # Easier to use the inverted lookup table in this context.
    inv_lt = invert_lookup(lt)

    sample = ""
    idx = 0
    while True:
        p = n_bigram[idx]
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()

        # Append the chars based on lookup table.
        sample += inv_lt[idx]
        if idx == 0:
            break

    return sample


def softmax(x):
    """Softmax activation function."""
    return x.exp() / x.exp().sum(1, keepdim=True)


def nll(words: List[str], n_bigram: torch.Tensor, lt: Dict[str, int]):
    """Compute the negative log likelihood of some input words based on bigram."""
    log_l = 0.0
    n = 0
    for w in words:
        aw = ["."] + list(w) + ["."]
        for c1, c2 in zip(aw, aw[1:]):
            log_l += torch.log(n_bigram[lt[c1]][lt[c2]])
            n += 1

    return -log_l / n


def train(xs, ys, w, lr: float = 0.1, reg_w: float = 0.01):
    """Performing optimization."""
    for _ in range(50):
        # Forward pass.
        logits = torch.matmul(xs, w)
        p = softmax(logits)
        loss = -p[torch.arange(len(xs)), ys].log().mean() + reg_w * (w**2).mean()
        print(f"loss: {loss.item():.3f}")

        # Backward pass.
        w.grad = None
        loss.backward()

        # update
        w.data += -lr * w.grad


def vis_bigram(bigram, lt):
    """Visualize Bigram via 2D plot."""
    # Easier to use the inverted lookup table in this context.
    inv_lt = invert_lookup(lt)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 16))
    plt.imshow(bigram, cmap="Blues")

    for i in range(bigram.shape[0]):
        for j in range(bigram.shape[1]):
            plt.text(j, i, lt[i] + lt[j], ha="center", va="bottom", color="gray")
            plt.text(j, i, bigram[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    with open("names.txt", "r") as f:
        words = f.read().splitlines()
        # Compute the lookup table.
        lt = def_lookup_table(words)
        # Compute the Bigram.
        bgr = bigram(words, lt)

        # Visualize bigram matrix.
        # vis_bigram(bgr, words)

        # Normalize the bigram along the rows.
        n_bgr = norm_bigram(bgr)

        # Draw a new sample from bigram.
        # set_of_words = [sample_from_bigram(n_bgr, lt, seed=i) for i in range(10)]

        # Create training data.
        xs, ys = create_dataset(words, lt)
        # Input weights.
        w = torch.randn((len(lt), len(lt)), requires_grad=True)

        # Perform training.
        train(xs, ys, w, lr=50)
