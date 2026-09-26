"""PyTorch reference attention and a hand-derived backward pass.

Forward:
    S = Q K^T * scale
    P = softmax(S)
    O = P V

Backward (given dO = dL/dO):
    dV = P^T dO
    dP = dO V^T
    dS = P * (dP - rowsum(P * dP))      # softmax backward, no N x N Jacobian
    dQ = dS K * scale
    dK = dS^T Q * scale

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


def attention_backward_reference(Q, K, V, dO, scale=None, causal=False):
    """Naive backward that materialises S, P, dP and dS. Returns (dQ, dK, dV)."""
    if scale is None:
        scale = default_scale(Q.shape[-1])

    # Recompute forward intermediates
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        mask = _causal_mask(Q.shape[-2], Q.device)
        S = S.masked_fill(mask, float("-inf"))
    P = torch.softmax(S, dim=-1)

    # O = P V
    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))

    # P = softmax(S), row-wise: dS_i = P_i * (dP_i - sum_j P_j dP_j)
    dS = P * (dP - torch.sum(dP * P, dim=-1, keepdim=True))
    if causal:
        # Masked positions did not participate in the forward pass
        dS = dS.masked_fill(mask, 0.0)

    # S = Q K^T * scale
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
    return dQ, dK, dV
