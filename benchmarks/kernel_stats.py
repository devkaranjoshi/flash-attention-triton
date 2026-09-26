"""Per-kernel GPU time (CUPTI trace via torch.profiler) and compiled resource
usage (registers, spills, shared memory, warps) for every Triton kernel.

Works without GPU performance-counter permissions, unlike Nsight Compute.

Usage:  python benchmarks/kernel_stats.py [--n 1024 4096] [--out stats.json]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flash_attention import (  # noqa: E402
    attention_backward_triton,
    attention_forward_triton,
    flash_attention_forward,
)
from flash_attention import naive_triton, online_forward  # noqa: E402

B, H, D = 1, 8, 64
ITERS = 10

KERNELS = {
    "online_softmax_attention": online_forward.online_softmax_attention,
    "forward_score_kernel": naive_triton.forward_score_kernel,
    "forward_softmax_kernel": naive_triton.forward_softmax_kernel,
    "forward_output_kernel": naive_triton.forward_output_kernel,
    "backward_dp_kernel": naive_triton.backward_dp_kernel,
    "backward_ds_kernel": naive_triton.backward_ds_kernel,
    "backward_dq_kernel": naive_triton.backward_dq_kernel,
    "backward_dk_kernel": naive_triton.backward_dk_kernel,
    "backward_dv_kernel": naive_triton.backward_dv_kernel,
}


def compiled_info(jit_fn):
    """Resource usage of the most recently compiled specialisation of a kernel."""
    out = []
    for cache in jit_fn.device_caches.values():
        for kernel in cache[0].values():
            md = kernel.metadata
            out.append({
                "regs": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared_bytes": md.shared,
                "num_warps": md.num_warps,
                "num_stages": md.num_stages,
            })
    return out


def run(N):
    torch.manual_seed(0)
    Q, K, V, dO = (torch.randn(B, H, N, D, device="cuda") for _ in range(4))
    q0 = (Q[0, 0] / math.sqrt(D)).contiguous()
    k0, v0 = K[0, 0].contiguous(), V[0, 0].contiguous()

    def step():
        flash_attention_forward(q0, k0, v0)
        attention_forward_triton(Q, K, V)
        attention_backward_triton(Q, K, V, dO)

    step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(ITERS):
            step()
        torch.cuda.synchronize()

    # The backward re-runs the score + softmax kernels: separate those launches
    times = defaultdict(list)
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.name in KERNELS:
            times[evt.name].append(evt.device_time)

    result = {}
    for name in KERNELS:
        t = times.get(name, [])
        launches_per_step = len(t) // ITERS
        result[name] = {
            "launches_per_step": launches_per_step,
            "mean_us": sum(t) / len(t) if t else None,
            "compiled": compiled_info(KERNELS[name]),
        }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1024, 4096])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"gpu": torch.cuda.get_device_name(0), "B": B, "H": H, "D": D, "runs": {}}
    for N in args.n:
        stats = run(N)
        report["runs"][N] = stats
        print(f"\nN={N}  (B={B}, H={H}, D={D}; online kernel is one head)")
        print(f"{'kernel':<26} {'launch/step':>11} {'mean us':>10} {'regs':>5} {'spills':>6} {'smem B':>7} {'warps':>5}")
        for name, s in stats.items():
            c = s["compiled"][-1] if s["compiled"] else {}
            mean = f"{s['mean_us']:.1f}" if s["mean_us"] else "-"
            print(f"{name:<26} {s['launches_per_step']:>11} {mean:>10} {c.get('regs', '-'):>5} "
                  f"{c.get('spills', '-'):>6} {c.get('shared_bytes', '-'):>7} {c.get('num_warps', '-'):>5}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
