"""Naive multi-head attention forward in Triton.

Shapes: Q, K, V, dO, O are [B, H, N, D]; D must be a power of two.

Forward:
    S = Q K^T * scale
    P = softmax(S)
    O = P V

This is a correctness reference, NOT the FlashAttention algorithm: it
materialises S, P, dP and dS as [B, H, N, N] float32 tensors in global
memory, with one small kernel per stage of the computational graph.
Batch and head are flattened onto one grid axis (b = pid // H, h = pid % H).
"""

import math

import torch
import triton
import triton.language as tl


def cdiv(x, y):
    return (x + y - 1) // y


def next_power_of_2(x):
    return 1 << (x - 1).bit_length()


# ============================================================
#
#                       FORWARD
#
# ============================================================


# ============================================================
# FORWARD 1
#
# S = Q K^T * scale
#
# Q : [B,H,N,D]
# K : [B,H,N,D]
# S : [B,H,N,N]
#
# Triton program:
#
#     program_id(0) -> query block
#     program_id(1) -> batch * head
#     program_id(2) -> key block
#
# ============================================================

@triton.jit
def forward_score_kernel(
    Q,
    K,
    S,

    stride_qb,
    stride_qh,
    stride_qn,
    stride_qd,

    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,

    stride_sb,
    stride_sh,
    stride_si,
    stride_sj,

    H,
    scale,

    N: tl.constexpr,
    D: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_n = tl.program_id(2)

    # --------------------------------------------------------
    # Flatten batch and head into one program dimension
    # --------------------------------------------------------

    b = pid_bh // H
    h = pid_bh % H

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    # Key tile owned by this program (grid axis 2)
    offs_n = (
        pid_n * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )

    offs_d = tl.arange(
        0,
        D,
    )

    # --------------------------------------------------------
    # Load Q
    # --------------------------------------------------------

    q_ptrs = (
        Q
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qn
        + offs_d[None, :] * stride_qd
    )

    q = tl.load(
        q_ptrs,
        mask=(
            (offs_m[:, None] < N)
            &
            (offs_d[None, :] < D)
        ),
        other=0.0,
    )

    # --------------------------------------------------------
    # Load K
    # --------------------------------------------------------

    k_ptrs = (
        K
        + b * stride_kb
        + h * stride_kh
        + offs_n[:, None] * stride_kn
        + offs_d[None, :] * stride_kd
    )

    k = tl.load(
        k_ptrs,
        mask=(
            (offs_n[:, None] < N)
            &
            (offs_d[None, :] < D)
        ),
        other=0.0,
    )

    # --------------------------------------------------------
    # QK^T
    # --------------------------------------------------------

    s = tl.dot(
        q,
        tl.trans(k),
        input_precision="ieee",
    )

    # Scale attention scores
    s = s * scale

    # --------------------------------------------------------
    # Mask invalid K positions
    # --------------------------------------------------------

    s = tl.where(
        offs_n[None, :] < N,
        s,
        -float("inf"),
    )

    # --------------------------------------------------------
    # Store S
    # --------------------------------------------------------

    s_ptrs = (
        S
        + b * stride_sb
        + h * stride_sh
        + offs_m[:, None] * stride_si
        + offs_n[None, :] * stride_sj
    )

    tl.store(
        s_ptrs,
        s,
        mask=(
            (offs_m[:, None] < N)
            &
            (offs_n[None, :] < N)
        ),
    )


# ============================================================
# FORWARD 2
#
# P = softmax(S)
#
# One program = one row.
# ============================================================

@triton.jit
def forward_softmax_kernel(
    S,
    P,

    stride_sb,
    stride_sh,
    stride_si,
    stride_sj,

    stride_pb,
    stride_ph,
    stride_pi,
    stride_pj,

    H,

    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    row = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_n = tl.arange(
        0,
        BLOCK_N,
    )

    # --------------------------------------------------------
    # Load S row
    # --------------------------------------------------------

    s_ptrs = (
        S
        + b * stride_sb
        + h * stride_sh
        + row * stride_si
        + offs_n * stride_sj
    )

    s = tl.load(
        s_ptrs,
        mask=offs_n < N,
        other=-float("inf"),
    )

    # --------------------------------------------------------
    # Stable softmax
    # --------------------------------------------------------

    row_max = tl.max(
        s,
        axis=0,
    )

    exp_s = tl.exp(
        s - row_max,
    )

    denom = tl.sum(
        exp_s,
        axis=0,
    )

    p = exp_s / denom

    # --------------------------------------------------------
    # Store P
    # --------------------------------------------------------

    p_ptrs = (
        P
        + b * stride_pb
        + h * stride_ph
        + row * stride_pi
        + offs_n * stride_pj
    )

    tl.store(
        p_ptrs,
        p,
        mask=offs_n < N,
    )


# ============================================================
# FORWARD 3
#
# O = P V
#
# P : [B,H,N,N]
# V : [B,H,N,D]
# O : [B,H,N,D]
#
# This kernel loops over K/V tiles.
# ============================================================

@triton.jit
def forward_output_kernel(
    P,
    V,
    O,

    stride_pb,
    stride_ph,
    stride_pi,
    stride_pj,

    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,

    stride_ob,
    stride_oh,
    stride_oi,
    stride_od,

    H,

    N: tl.constexpr,
    D: tl.constexpr,

    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = (
        pid_m * BLOCK_M
        + tl.arange(0, BLOCK_M)
    )

    offs_n = tl.arange(
        0,
        BLOCK_N,
    )

    offs_d = tl.arange(
        0,
        D,
    )

    acc = tl.zeros(
        [BLOCK_M, D],
        dtype=tl.float32,
    )

    # --------------------------------------------------------
    # Loop over all K/V tiles
    # --------------------------------------------------------

    for start_n in range(
        0,
        N,
        BLOCK_N,
    ):

        n = (
            start_n
            + offs_n
        )

        # ----------------------------------------------------
        # Load P tile
        # ----------------------------------------------------

        p = tl.load(
            P
            + b * stride_pb
            + h * stride_ph
            + offs_m[:, None] * stride_pi
            + n[None, :] * stride_pj,

            mask=(
                (offs_m[:, None] < N)
                &
                (n[None, :] < N)
            ),

            other=0.0,
        )

        # ----------------------------------------------------
        # Load V tile
        # ----------------------------------------------------

        v = tl.load(
            V
            + b * stride_vb
            + h * stride_vh
            + n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,

            mask=(
                (n[:, None] < N)
                &
                (offs_d[None, :] < D)
            ),

            other=0.0,
        )

        # ----------------------------------------------------
        # P @ V
        # ----------------------------------------------------

        acc += tl.dot(
            p,
            v,
            input_precision="ieee",
        )

    # --------------------------------------------------------
    # Store O
    # --------------------------------------------------------

    o_ptrs = (
        O
        + b * stride_ob
        + h * stride_oh
        + offs_m[:, None] * stride_oi
        + offs_d[None, :] * stride_od
    )

    tl.store(
        o_ptrs,
        acc,
        mask=(
            (offs_m[:, None] < N)
            &
            (offs_d[None, :] < D)
        ),
    )


# ============================================================
# FORWARD WRAPPER
# ============================================================

def attention_forward_triton(
    Q,
    K,
    V,
    scale=None,
):

    B, H, N, D = Q.shape

    assert K.shape == Q.shape
    assert V.shape == Q.shape

    assert Q.is_cuda
    assert K.is_cuda
    assert V.is_cuda

    # --------------------------------------------------------
    # Allocate intermediates
    # --------------------------------------------------------

    S = torch.empty(
        (B, H, N, N),
        device=Q.device,
        dtype=torch.float32,
    )

    P = torch.empty_like(S)

    O = torch.empty(
        (B, H, N, D),
        device=Q.device,
        dtype=Q.dtype,
    )

    # --------------------------------------------------------
    # Block configuration
    #
    # BLOCK_N is a fixed key tile: the score kernel is launched
    # over a (query block, batch*head, key block) grid, so N is
    # not limited by shared memory.
    #
    # ROW_BLOCK covers a whole row for the row-wise softmax.
    # --------------------------------------------------------

    BLOCK_M = 16

    BLOCK_N = 64

    ROW_BLOCK = next_power_of_2(N)

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    # --------------------------------------------------------
    # S = QK^T
    # --------------------------------------------------------

    forward_score_kernel[
        (
            cdiv(N, BLOCK_M),
            B * H,
            cdiv(N, BLOCK_N),
        )
    ](
        Q,
        K,
        S,

        Q.stride(0),
        Q.stride(1),
        Q.stride(2),
        Q.stride(3),

        K.stride(0),
        K.stride(1),
        K.stride(2),
        K.stride(3),

        S.stride(0),
        S.stride(1),
        S.stride(2),
        S.stride(3),

        H=H,
        scale=scale,

        N=N,
        D=D,

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # --------------------------------------------------------
    # P = softmax(S)
    # --------------------------------------------------------

    forward_softmax_kernel[
        (
            N,
            B * H,
        )
    ](
        S,
        P,

        S.stride(0),
        S.stride(1),
        S.stride(2),
        S.stride(3),

        P.stride(0),
        P.stride(1),
        P.stride(2),
        P.stride(3),

        H=H,

        N=N,
        BLOCK_N=ROW_BLOCK,
    )

    # --------------------------------------------------------
    # O = PV
    # --------------------------------------------------------

    forward_output_kernel[
        (
            cdiv(N, BLOCK_M),
            B * H,
        )
    ](
        P,
        V,
        O,

        P.stride(0),
        P.stride(1),
        P.stride(2),
        P.stride(3),

        V.stride(0),
        V.stride(1),
        V.stride(2),
        V.stride(3),

        O.stride(0),
        O.stride(1),
        O.stride(2),
        O.stride(3),

        H=H,

        N=N,
        D=D,

        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return O