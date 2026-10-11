# FlashAttention in Triton: forward pass

Attention kernels written in [Triton](https://triton-lang.org), built step by step
from the computational graph and validated against PyTorch autograd:

```
S = Q Kᵀ · scale        dV = Pᵀ dO
P = softmax(S)          dP = dO Vᵀ
O = P V                 dS = P ⊙ (dP − rowsum(P ⊙ dP))
                        dQ = dS K · scale
                        dK = dSᵀ Q · scale
```

| Module | Description |
|--------|-------------|
| [`flash_attention/online_forward.py`](flash_attention/online_forward.py) | **Fused forward** with online softmax: each program owns a query tile and streams over K/V tiles, keeping a running max and sum, so the N×N matrix never touches global memory. |
| [`flash_attention/reference.py`](flash_attention/reference.py) | PyTorch reference forward |

## Results on an NVIDIA L4

All 47 correctness tests pass (N up to 4096, D up to 128). Full details are in the
**[evaluation report](docs/REPORT.md)**.

| N (B=1, H=8, D=64) | Naive fwd | Naive bwd | PyTorch eager fwd+bwd | PyTorch SDPA fwd+bwd | Naive bwd memory |
|---|---|---|---|---|---|
| 1024 | 0.70 ms | 7.1 ms | 2.8 ms | 1.3 ms | 134 MiB |
| 4096 | 15.0 ms | 125.5 ms | 47.5 ms | 18.1 ms | 2072 MiB |

The fused forward uses 24 MiB at N = 4096, versus 1 GiB for the naive forward.

## Usage

```python
import torch
from flash_attention import (
    attention_forward_triton,  flash_attention_forward,
)

Q, K, V, dO = (torch.randn(2, 8, 1024, 64, device="cuda") for _ in range(4))

O = attention_forward_triton(Q, K, V)                 # [B, H, N, D], scale = 1/sqrt(D)

o = flash_attention_forward(Q[0, 0], K[0, 0], V[0, 0])  # fused, single head, unscaled
```

## Running

```bash
pip install -r requirements.txt   # torch, triton, pytest; needs an NVIDIA GPU

pytest                                         # correctness vs PyTorch autograd
python benchmarks/bench.py                     # latency + peak memory vs PyTorch
python benchmarks/kernel_stats.py              # per-kernel time, registers, shared memory
bash benchmarks/profile_ncu.sh                 # Nsight Compute (needs GPU counter access)
```

`tests/test_reference.py` also runs on CPU.

## Limitations

- float32 only; D must be a power of two; no causal mask in the Triton kernels.
- The naive kernels use O(B·H·N²) memory by design.
- The fused forward is single-head and unscaled (scale Q beforehand).

## Roadmap

1. Tile-level `tl.dot` for dQ / dK / dV (coalesced access, tensor cores).
2. `[B, H, N, D]` fused forward that saves the row log-sum-exp.