import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def autograd_grads(fn, Q, K, V, dO):
    """dQ, dK, dV of sum(fn(Q, K, V) * dO), i.e. backprop of upstream gradient dO."""
    Q, K, V = (t.detach().clone().requires_grad_(True) for t in (Q, K, V))
    (fn(Q, K, V) * dO).sum().backward()
    return Q.grad, K.grad, V.grad
