"""Small Layers Library"""
import torch
import torch.nn.functional as F


class Linear:
    def __init__(self, in_dim, out_dim, bias: bool = True):
        self.bias = bias
        self.w = torch.randn((in_dim, out_dim)) / (in_dim**0.5)
        self.b = torch.zeros(out_dim) if bias else None

    def __call__(self, input):
        self.out = input @ self.w
        if self.b is not None:
            self.out += self.b
        return self.out

    def params(self):
        return [self.w] + ([] if self.b is None else [self.b])


class BatchNorm1D():
    def __init__(self, size: int, eps: float = 1e-5, momentum: float = 0.1, train: bool = True):
        self.eps = eps
        self.momentum = momentum
        self.train = train
        # trainable
        self.bngain = torch.ones((1, size))
        self.bnbias = torch.zeros((1, size))
        # non-trainable
        self.bnmean = torch.zeros((1, size))
        self.bnstd = torch.ones((1, size))

    def __call__(self, input):
        if self.train:
            bnmean_i = input.mean(0, keepdims=True)
            bnstd_i = input.std(0, keepdims=True)
        else:
            bnmean_i = self.bnmean
            bnstd_i = self.bnstd

        input_hat = (input - bnmean_i) / torch.sqrt(bnstd_i + self.eps)
        self.out = self.bngain * input_hat / self.bnbias

        if self.train:
            with torch.no_grad():
                self.bnmean = (
                    1 - self.momentum
                ) * self.bnmean + self.momentum * bnmean_i
                self.bnstd = (1 - self.momentum) * self.bnstd + self.momentum * bnstd_i

        return self.out

    def params(self):
        return [self.bngain, self.bnbias]


class Tanh():
    def __call__(self, input):
        self.out = torch.tanh(input)
        return self.out

    def params(self):
        return []


class Embeddings():
    """Check out torch.nn.Embedding for reference"""

    def __init__(self, num_embeddings, emb_dims):
        self.w = torch.randn((num_embeddings, emb_dims))
        print("emb :", self.w.shape)

    def __call__(self, input_idx):
        self.out = self.w[input_idx]
        return self.out

    def params(self):
        return [self.w]


class Flatten():
    def __call__(self, input):
        self.out = input.view(input.shape[0], -1)
        return self.out

    def params(self):
        return []


class Sequential():
    def __init__(self, layers: list):
        self.layers = layers

    def __call__(self, input):
        for layer in self.layers:
            input = layer(input)
        self.out = input
        return self.out

    def params(self):
        return [p for layer in self.layers for p in layer.params() if type(p) != bool]
