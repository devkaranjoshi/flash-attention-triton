"""PyTorch reference attention.

Forward:
    S = Q K^T * scale
    P = softmax(S)
    O = P V

Shapes: Q, K, V, dO are [B, H, N, D] (any leading batch dims work).
"""

import torch


def default_scale(head_dim):
    return 1.0 / (head_dim ** 0.5)


def _causal_mask(n, device):
    return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)


def attention_reference(Q, K, V, scale=None, causal=False):
    """Standard attention forward. Used as the autograd oracle in tests."""
    if scale is None:
        scale = default_scale(Q.shape[-1])

    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        S = S.masked_fill(_causal_mask(Q.shape[-2], Q.device), float("-inf"))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)
