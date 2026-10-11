"""Latency and peak-memory benchmark on the current GPU.

Compares, for each sequence length N:
  * online forward      - fused online-softmax kernel (one head, looped over B*H heads)
  * naive Triton fwd - materialises [B,H,N,N] S / P / dP / dS
  * PyTorch eager        - matmul + softmax + matmul
  * PyTorch SDPA         - F.scaled_dot_product_attention (fused)

Usage:  python benchmarks/bench.py [--out results.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flash_attention import (  # noqa: E402
    attention_forward_triton,
    attention_reference,
    flash_attention_forward,
)

B, H, D = 1, 8, 64
SEQ_LENS = [256, 512, 1024, 2048, 4096]


def measure(fn):
    """(median ms, peak MiB above the baseline allocation)."""
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    ms = triton.testing.do_bench(fn, warmup=10, rep=50)
    return ms, peak


def online_forward_all_heads(Q, K, V):
    # The fused kernel is single-head [N, D] and unscaled: fold the scale into Q.
    Qs = Q * (1.0 / math.sqrt(D))
    return torch.stack([
        flash_attention_forward(Qs[b, h], K[b, h], V[b, h])
        for b in range(B) for h in range(H)
    ]).view(B, H, -1, D)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)} | B={B} H={H} D={D} | float32\n")
    header = f"{'N':>6} | {'case':<22} | {'ms':>9} | {'peak MiB':>9}"
    print(header)
    print("-" * len(header))

    rows = []
    for N in SEQ_LENS:
        torch.manual_seed(0)
        Q, K, V, dO = (torch.randn(B, H, N, D, device="cuda") for _ in range(4))
        cases = {
            "online forward": lambda: online_forward_all_heads(Q, K, V),
            "naive Triton fwd": lambda: attention_forward_triton(Q, K, V),
            "PyTorch eager fwd": lambda: attention_reference(Q, K, V),
            "PyTorch SDPA fwd": lambda: F.scaled_dot_product_attention(Q, K, V)
        }
        for name, fn in cases.items():
            ms, mib = measure(fn)
            rows.append({"N": N, "case": name, "ms": ms, "peak_mib": mib})
            print(f"{N:>6} | {name:<22} | {ms:>9.3f} | {mib:>9.1f}")
        print()
        del Q, K, V, dO
        torch.cuda.empty_cache()

    if args.out:
        Path(args.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "B": B, "H": H, "D": D, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
