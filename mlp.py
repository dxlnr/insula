"""Multi-Layer-Perceptron"""
import torch
import torch.nn.functional as F

# Hyperparameters
EMBEDDING_SPACE = 10
BLOCK_SIZE = 3
BATCH_SIZE = 16


def def_lookup_table(words: list[str]):
    """Returns a lookup table with all the chars in words plus special char
    for start and ending."""
    clist = ["."] + sorted(list(set("".join(words))))
    return {s: i for i, s in enumerate(clist)}


def get_dataset(words: list[str], lt: dict[str, int], block_size: int = 3):
    x, y = [], [] 
    for w in words:

        context = [0] * BLOCK_SIZE
        for i, ch in enumerate(w + '.'):
            x.append(context)
            y.append(lt[ch])
            context = context[1:] + [lt[ch]]

    return torch.Tensor(x), torch.Tensor(y) 


def build_model(hidden_layers: int = 100):
    c = torch.randn((len(lt), EMBEDDING_SPACE))

    w1 = torch.randn((BLOCK_SIZE * EMBEDDING_SPACE, hidden_layers))
    b1 = torch.randn(hidden_layers)

    w2 = torch.randn((hidden_layers, len(lt)))
    b2 = torch.randn(len(lt))
    
    return [c, w1, b1, w2, b2]


def generate_sequence(params, n: int=5):
    for _ in range(n):
        out = []
        context = [0] * BLOCK_SIZE

        while True:
            emb = params[0][torch.tensor([context])]
            h = torch.tanh(emb.view(-1, BLOCK_SIZE*EMBEDDING_SPACE) @ params[1] + params[2])
            logits = h @ params[3] + params[4]
            probs = F.softmax(logits, dim=1)

            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            out.append(ix)

            if ix == 0:
                break
           
        print(''.join(list(lt.keys())[list(lt.values()).index(i)] for i in out))


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

        for p in params:
            p.requires_grad = True
        
        lr = 0.1
        for i in range(100000):
            b_idx = torch.randint(0, x.shape[0], (BATCH_SIZE, ))

            # forward pass
            embeddings = params[0][x[b_idx]]
            h = torch.tanh(embeddings.view(-1, BLOCK_SIZE*EMBEDDING_SPACE) @ params[1] + params[2])
            logits = h @ params[3] + params[4]
            
            loss = F.cross_entropy(logits, labels[b_idx])
            
            # backward pass
            for p in params:
                p.grad = None
            loss.backward()

            # update
            for p in params:
                p.data += -lr * p.grad
            
            if i % 10000 == 0:
                print(loss.item())

            if i > 25000:
                lr = 0.01

        print("")
        print("final loss : ", loss.item())
        print("") 
        generate_sequence(params)
