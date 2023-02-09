"""Multi-Layer-Perceptron"""
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
