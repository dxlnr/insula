"""Multi-Layer-Perceptron"""
import random

import torch
import torch.nn.functional as F

# Hyperparameters
EMBEDDING_SPACE = 12
BLOCK_SIZE = 3
HIDDEN_LAYERS = 500
BATCH_SIZE = 32


def def_lookup_table(words: list[str]):
    """Returns a lookup table with all the chars in words plus special char
    for start and ending."""
    clist = ["."] + sorted(list(set("".join(words))))
    return {s: i for i, s in enumerate(clist)}


def get_dataset(words: list[str], lt: dict[str, int], block_size: int = 3):
    x, y = [], []
    for w in words:

        context = [0] * BLOCK_SIZE
        for i, ch in enumerate(w + "."):
            x.append(context)
            y.append(lt[ch])
            context = context[1:] + [lt[ch]]

    return torch.Tensor(x), torch.Tensor(y)


def split_dataset(words):
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    xtr, ytr = get_dataset(words[:n1])  # 80%
    xval, yval = get_dataset(words[n1:n2])  # 10%
    xte, yte = get_dataset(words[n2:])  # 10%

    return xtr, ytr, xval, yval, xte, yte


def build_model():
    c = torch.randn((len(lt), EMBEDDING_SPACE))

    w1 = (
        torch.randn((BLOCK_SIZE * EMBEDDING_SPACE, HIDDEN_LAYERS))
        * (5 / 3)
        / ((BLOCK_SIZE * EMBEDDING_SPACE) ** 0.5)
    )
    # BatchNormalization Layer
    bngain = torch.ones((1, HIDDEN_LAYERS))
    bnbias = torch.zeros((1, HIDDEN_LAYERS))
    bn_mean = torch.zeros((1, HIDDEN_LAYERS))
    bn_std = torch.ones((1, HIDDEN_LAYERS))

    w2 = torch.randn((HIDDEN_LAYERS, len(lt))) * 0.01
    b2 = torch.randn(len(lt)) * 0.0

    # Return Model with [trainable params] & [non-trainable params] in terms of
    # the gradient.
    return [[c, w1, bngain, bnbias, w2, b2], [bn_mean, bn_std]]


def forward(xb, yb, c, w1, b1, bngain, bnbias, w2, b2):
    emb = c[xb]  # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1)  # concatenate the vectors

    # Linear layer 1
    hprebn = embcat @ w1 + b1  # hidden layer pre-activation

    # BatchNorm layer
    bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiff2 = bndiff**2
    bnvar = (
        1 / (n - 1) * (bndiff2).sum(0, keepdim=True)
    )  # note: Bessel's correction (dividing by n-1, not n)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias

    # Non-linearity
    h = torch.tanh(hpreact)  # hidden layer
    # Linear layer 2
    logits = h @ w2 + b2  # output layer

    # cross entropy loss (same as F.cross_entropy(logits, Yb))
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes  # subtract max for numerical stability
    counts = norm_logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum**-1
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), yb].mean()

    return loss


def backward():
    d_logprobs = torch.zeros_like(logprobs)
    d_logprobs[range(n), yb] = -1.0/n

    # cross-entropy-loss
    d_probs = d_logprobs * (1 / probs)
    d_counts_sum_inv = (d_probs * counts).sum(1, keepdim=True)
    d_counts_sum = d_counts_sum_inv * (-counts_sum**-2)
    d_counts = counts_sum_inv * d_probs
    d_counts += d_counts_sum * torch.ones_like(counts) 
    d_norm_logits = d_counts * norm_logits.exp()
    d_logit_maxes = (-d_norm_logits).sum(1, keepdim=True)

    # Linear layer 2
    d_logits = d_norm_logits.clone()
    d_logits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * d_logit_maxes

    # Non-Linearity
    d_h = d_logits @ w2.T
    d_W2 = h.T @ d_logits
    d_b2 = d_logits.sum(0)

    # BatchNorm layer
    d_hpreact = d_h * (1 - torch.tanh(hpreact)**2)
    d_bnraw = d_hpreact * bngain
    d_bnvar_inv = (d_bnraw * bndiff).sum(0, keepdim=True)
    d_bnvar = d_bnvar_inv * (-0.5 * (bnvar + 1e-5)**-1.5)
    d_bndiff2 = d_bnvar * (1.0/(n-1))*torch.ones_like(bndiff2)
    d_bndiff = d_bnraw * bnvar_inv
    d_bndiff += 2*bndiff * d_bndiff2
    d_bnmeani = (-d_bndiff).sum(0, keepdim=True)
    d_hprebn = d_bndiff.clone()
    d_hprebn += 1.0/n * (torch.ones_like(hprebn) * d_bnmeani)

    # Linear Layer1
    d_embcat = d_hprebn @ w1.T

    d_w1 = embcat.T @ d_hprebn
    d_emb = d_embcat.view(emb.shape)
    d_c = torch.zeros_like(c)
    for k in range(xb.shape[0]):
      for j in range(xb.shape[1]):
        ix = xb[k,j]
        d_c[ix] += d_emb[k,j]

    return d_c, d_w1, d_w2, d_b2

def generate_sequence(params, n: int = 5):
    for _ in range(n):
        out = []
        context = [0] * BLOCK_SIZE

        while True:
            emb = params[0][0][torch.tensor([context])]
            hp = torch.tanh(emb.view(-1, BLOCK_SIZE * EMBEDDING_SPACE) @ params[0][1])

            hp = params[0][2] * (hp - params[1][0]) / (params[0][3] + params[1][1])

            logits = hp @ params[0][4] + params[0][5]
            probs = F.softmax(logits, dim=1)

            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)

            if ix == 0:
                break

        print("".join(list(lt.keys())[list(lt.values()).index(i)] for i in out))


if __name__ == "__main__":
    global lt, words

    with open("names.txt", "r") as f:
        words = f.read().splitlines()
        lt = def_lookup_table(words)

        # Construct the dataset.
        x, labels = get_dataset(words, lt)
        x = x.long()
        labels = labels.long()

        params = build_model()

        for p in params[0]:
            p.requires_grad = True

        for i in range(120000):
            b_idx = torch.randint(0, x.shape[0], (BATCH_SIZE,))

            # forward pass
            embeddings = params[0][0][x[b_idx]]
            hp = torch.tanh(
                embeddings.view(-1, BLOCK_SIZE * EMBEDDING_SPACE) @ params[0][1]
            )
            # BatchNorm Layer
            bnmean_i = hp.mean(0, keepdims=True)
            bnstd_i = hp.std(0, keepdims=True)
            hp = params[0][2] * (hp - bnmean_i) / bnstd_i + params[0][3]

            with torch.no_grad():
                params[1][0] = 0.999 * params[1][0] + 0.001 * bnmean_i
                params[1][1] = 0.999 * params[1][1] + 0.001 * bnstd_i

            logits = hp @ params[0][4] + params[0][5]

            loss = F.cross_entropy(logits, labels[b_idx])

            # backward pass
            for p in params[0]:
                p.grad = None
            loss.backward()

            lr = 0.1 if i < 10000 else 0.01
            # update
            for p in params[0]:
                p.data += -lr * p.grad

            if i % 10000 == 0:
                print(f"{loss.item():.4f}")

        print("")
        print("final loss : ", loss.item())
        print("")
        print("Results: ")
        generate_sequence(params)
