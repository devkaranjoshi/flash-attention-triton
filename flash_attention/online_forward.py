"""Fused FlashAttention-style forward pass with online softmax (single head).

One Triton program owns a BLOCK_M tile of queries and streams over K/V in
BLOCK_N tiles, keeping a running row max ``m_i`` and running denominator
``l_i``. The N x N score matrix is never written to global memory.

Computes O = softmax(Q K^T) V with NO 1/sqrt(D) scaling, for Q, K, V of shape
[N, D] (float32). D must be a power of two.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_attention(
    Q, K, V, O,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vn, stride_vk,
    stride_om, stride_ok,
    N_CTX: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Q: [BLOCK_M, D]
    q = tl.load(
        Q
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qk,
        mask=offs_m[:, None] < N_CTX,
        other=0.0,
    )

    # Running online-softmax state
    m_i = tl.full(
        [BLOCK_M],
        -float("inf"),
        dtype=tl.float32,
    )

    l_i = tl.zeros(
        [BLOCK_M],
        dtype=tl.float32,
    )

    acc = tl.zeros(
        [BLOCK_M, D],
        dtype=tl.float32,
    )

    for start_n in range(0, N_CTX, BLOCK_N):

        offs_n = start_n + tl.arange(0, BLOCK_N)

        # ------------------------------------------------
        # Load K
        # ------------------------------------------------

        k = tl.load(
            K
            + offs_n[:, None] * stride_kn
            + offs_d[None, :] * stride_kk,
            mask=(
                (offs_n[:, None] < N_CTX)
            ),
            other=0.0,
        )

        # ------------------------------------------------
        # Load V
        # ------------------------------------------------

        v = tl.load(
            V
            + offs_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vk,
            mask=(
                offs_n[:, None] < N_CTX
            ),
            other=0.0,
        )

        # ------------------------------------------------
        # QK^T
        # ------------------------------------------------

        qk = tl.dot(q, tl.trans(k), input_precision="ieee")

        # Mask invalid K positions
        qk = tl.where(
            offs_n[None, :] < N_CTX,
            qk,
            -float("inf"),
        )

        # ------------------------------------------------
        # Online softmax update
        # ------------------------------------------------

        m_tile = tl.max(qk, axis=1)

        m_new = tl.maximum(
            m_i,
            m_tile,
        )

        alpha = tl.exp(
            m_i - m_new
        )

        # Instead of explicitly computing beta:
        #
        # beta * exp(qk - m_tile)
        #
        # we directly compute:
        #
        # exp(qk - m_new)

        p = tl.exp(
            qk - m_new[:, None]
        )

        l_i = (
            alpha * l_i
            + tl.sum(p, axis=1)
        )

        acc = (
            alpha[:, None] * acc
            + tl.dot(p, v, input_precision="ieee")
        )

        m_i = m_new

    # ----------------------------------------------------
    # Final normalization
    # ----------------------------------------------------
    acc = acc / l_i[:, None]

    # ----------------------------------------------------
    # Store
    # ----------------------------------------------------
    tl.store(
        O
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_ok,
        acc,
        mask=offs_m[:, None] < N_CTX,
    )


def flash_attention_forward(Q, K, V, BLOCK_M=64, BLOCK_N=64):
    """Launch the online-softmax kernel. Q, K, V: [N, D] float32 CUDA tensors."""
    assert Q.is_cuda and K.is_cuda and V.is_cuda
    N, D = Q.shape
    assert K.shape == (N, D) and V.shape == (N, D)
    assert D & (D - 1) == 0, "D must be a power of two"

    O = torch.empty_like(Q)
    online_softmax_attention[(triton.cdiv(N, BLOCK_M),)](
        Q, K, V, O,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        N_CTX=N,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    return O
