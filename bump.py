"""Various Language Models."""
from typing import List 
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def def_lookup_table(words: List[str]):
    """Returns a lookup table with all the chars in words plus special char 
    for start and ending."""
    clist = ['.'] + sorted(list(set(''.join(words)))) 
    return {s:i for i, s in enumerate(clist)}


def bigram(words: List[str]):
    """Returns the Bigram of a list of str."""
    clt = def_lookup_table(words)
    bigram = np.zeros([len(clt), len(clt)], dtype=np.uint32)

    for w in words:
        aw = ['.'] + list(w) + ['.']
        for c1, c2 in zip(aw, aw[1:]):
            bigram[clt[c1]][clt[c2]] += 1

    return bigram

def vis_bigram(bigram, words):
    """Visualize Bigram via 2D plot."""
    clist = ['.'] + sorted(list(set(''.join(words)))) 
    lt = {i:s for i, s in enumerate(clist)}

    import matplotlib.pyplot as plt
    plt.figure(figsize=(16,16))
    plt.imshow(bigram, cmap="Blues")

    for i in range(bigram.shape[0]):
        for j in range(bigram.shape[1]):
            plt.text(j, i, lt[i] + lt[j], ha="center", va="bottom", color="gray")
            plt.text(j, i, bigram[i, j].item(), ha="center", va="top", color="gray")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    with open("names.txt", "r") as f:
        words = f.read().splitlines()
        # Compute the Bigram.
        bgr = bigram(words)
        # Visualize bigram matrix. 
        vis_bigram(bgr, words)
