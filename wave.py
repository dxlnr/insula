"""WaveNet"""
import torch
import torch.nn.functional as F

from data import def_lookup_table, get_dataset
from nn.layers import Embeddings, ConsFlatten, Sequential, Linear, BatchNorm1D, Tanh


# Hyperparameters
EMBEDDING_SPACE = 10
BLOCK_SIZE = 8
BLOCK_SPLIT = 2
STEPS = 100000
BATCH_SIZE = 32
N_HIDDEN = 250


def build_model(vocab_size: int):
    model = Sequential(
        [
            Embeddings(vocab_size, EMBEDDING_SPACE),
            ConsFlatten(BLOCK_SPLIT), Linear(EMBEDDING_SPACE * BLOCK_SPLIT, N_HIDDEN, bias=True), 
            BatchNorm1D(N_HIDDEN),
            Tanh(),
            ConsFlatten(BLOCK_SPLIT), Linear(N_HIDDEN * BLOCK_SPLIT, N_HIDDEN, bias=True),
            BatchNorm1D(N_HIDDEN),
            Tanh(),
            ConsFlatten(BLOCK_SPLIT), Linear(N_HIDDEN * BLOCK_SPLIT, N_HIDDEN, bias=True),
            BatchNorm1D(N_HIDDEN),
            Tanh(),
            Linear(N_HIDDEN, vocab_size),
        ]
    )

    for p in model.params():
        if p is not None:
            p.requires_grad = True
    
    return model


def inspect_model(model):
    for layer in model.layers:
        print(layer.__class__.__name__, ":", tuple(layer.out.shape))


def generate_sample(model, lt, n: int = 3, block_size: int = 8):
    """Generate Sample from model."""
    for _ in range(n):
        out = []
        context = [0] * block_size 
        while True:
          # forward pass the neural net
          logits = model(torch.tensor([context]))
          probs = F.softmax(logits, dim=1)
          # sample from the distribution
          ix = torch.multinomial(probs, num_samples=1).item()
          # shift the context window and track the samples
          context = context[1:] + [ix]
          out.append(ix)
          # if we sample the special '.' token, break
          if ix == 0:
            break
        
        print("".join(list(lt.keys())[list(lt.values()).index(i)] for i in out))


if __name__ == "__main__":
    global lt, words

    with open("names.txt", "r") as f:
        words = f.read().splitlines()
        lt = def_lookup_table(words)

        # Construct the dataset.
        x, y = get_dataset(words, lt, BLOCK_SIZE)
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
        
        print("")
        print("final loss : ", loss.item())
        print("")
        print("Results: ")
        generate_sample(model, lt, 5)
