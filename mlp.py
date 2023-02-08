"""Multi-Layer-Perceptron"""
import torch
import torch.nn.functional as F

# Hyperparameters
EMBEDDING_SPACE = 2
BLOCK_SIZE = 3


def def_lookup_table(words: list[str]):
    """Returns a lookup table with all the chars in words plus special char
    for start and ending."""
    clist = ["."] + sorted(list(set("".join(words))))
    return {s: i for i, s in enumerate(clist)}


def get_dataset(words: list[str], lt: dict[str, int], block_size: int = 3):
    x, y = [], [] 
    for w in words:
        for idx, ch in enumerate(w):
            if idx == 0:
                x.append([0, lt[ch], lt[w[idx+1]]])
                y.append(lt[w[idx+1]])
            elif idx == len(w) - 1:
                x.append([lt[w[idx-1]], lt[ch], 0])
                y.append(0)
            else:
                x.append([lt[w[idx-1]], lt[ch], lt[w[idx+1]]])
                y.append(lt[w[idx+1]])

    return torch.Tensor(x), torch.Tensor(y) 


def build_model(hidden_layers: int = 100):
    c = torch.randn((len(lt), EMBEDDING_SPACE))

    w1 = torch.randn((BLOCK_SIZE * EMBEDDING_SPACE, hidden_layers))
    b1 = torch.randn(hidden_layers)

    w2 = torch.randn((hidden_layers, len(lt)))
    b2 = torch.randn(len(lt))
    
    return [c, w1, b1, w2, b2]


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
        
        for i in range(250):
            # forward pass
            embeddings = params[0][x]
            h = torch.tanh(embeddings.view(-1, BLOCK_SIZE*EMBEDDING_SPACE) @ params[1] + params[2])
            logits = h @ params[3] + params[4]
            
            loss = F.cross_entropy(logits, labels)
            
            # backward pass
            for p in params:
                p.grad = None
            loss.backward()

            # update
            for p in params:
                p.data += -0.1 * p.grad
            
            if i % 50 == 0:
                print(loss.item())
        print(loss.item())
