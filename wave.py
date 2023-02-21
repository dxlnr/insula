"""WaveNet"""
import torch
import torch.nn.functional as F

from data import def_lookup_table, get_dataset
from nn.layers import Embeddings, Flatten, Sequential, Linear, BatchNorm1D, Tanh


# Hyperparameters
EMBEDDING_SPACE = 12
BLOCK_SIZE = 3
STEPS = 100000
BATCH_SIZE = 32
N_HIDDEN = 250


def build_model(vocab_size):
    model = Sequential(
        [
            Embeddings(vocab_size, EMBEDDING_SPACE),
            Flatten(),
            Linear(EMBEDDING_SPACE * BLOCK_SIZE, N_HIDDEN, bias=False),
            BatchNorm1D(N_HIDDEN),
            Tanh(),
            Linear(N_HIDDEN, vocab_size),
        ]
    )

    for p in model.params():
        if p is not None:
            p.requires_grad = True

    return model


if __name__ == "__main__":
    global lt, words

    with open("names.txt", "r") as f:
        words = f.read().splitlines()
        lt = def_lookup_table(words)

        # Construct the dataset.
        x, y = get_dataset(words, lt)
        x = x.long()
        y = y.long()

        vocab_size = len(lt)
        # Construct the model.
        model = build_model(vocab_size)

        for i in range(STEPS):
            # set up batches.
            ix = torch.randint(0, x.shape[0], (BATCH_SIZE,))
            xb, yb = x[ix], y[ix]

            # forward pass
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            # backward pass
            for p in model.params():
                p.grad = None
            loss.backward()

            lr = 0.1 if i < 10000 else 0.01
            # update
            for p in model.params():
                p.data += -lr * p.grad

            if i % 10000 == 0:
                print(f"{loss.item():.4f}")
