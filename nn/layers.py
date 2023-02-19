"""Small Layers Library"""
import torch


class Linear:
    def __init__(self, in_dim, out_dim, bias: bool = True):
        self.bias = bias
        self.w = torch.randn((in_dim, out_dim)) * (5 / 3) / (in_dim**0.5)
        b = torch.randn(out) * 0.0

    def __call__(self, input):
        out = input @ self.w
        if self.bias is not None:
            out += self.b
        return out

    def params(self):
        return [self.w] + ([] if self.bias is None else [self.bias])


class BatchNorm1D:
    def __init__(self, size: int, eps: float = 1e-5, momentum: float = 0.1):
        self.eps = eps
        self.momentum = momentum
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

    def params(self) -> list:
        return [self.bngain, self.bnbias]


class Tanh:
    def __call__(self, input):
        self.out = torch.tanh(input)
        return self.out

    def params(self):
        return []
