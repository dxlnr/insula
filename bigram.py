"""Various Language Models."""
from typing import List 
import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def char_in_str(str: str):
    clist = list()
    for i in str:
        if i not in clist:
            clist.append(i)
    clist.sort() 
    return clist


def bigram(words: List[str]):
    """Returns the Bigram of a list of str."""
    clist = char_in_str(''.join(words))
    bigram = np.zeros([len(clist), len(clist)])

    for w in words:
        for c1, c2 in zip(w, w[1:]):
            bigram[clist.index(c1)][clist.index(c2)] += 1

    return bigram / bigram.sum(), clist

if __name__ == "__main__":
    with open("names.txt", "r") as f:
        words = f.read().splitlines()

        bgr, clist = bigram(words)

        indices = np.where(bgr == bgr.max())
        print(f"Highest combination of chars in names is: {clist[indices[0][0]]}{clist[indices[1][0]]}")
