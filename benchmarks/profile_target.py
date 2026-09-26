"""Launch each kernel once at a fixed shape, for Nsight Compute.

    ncu --set full -o report python benchmarks/profile_target.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flash_attention import (  # noqa: E402
    attention_backward_triton,
    attention_forward_triton,
    flash_attention_forward,
)

B, H, N, D = 1, 8, 1024, 64

torch.manual_seed(0)
Q, K, V, dO = (torch.randn(B, H, N, D, device="cuda") for _ in range(4))

# Warm-up compiles every kernel so the profiled launches are steady-state
for _ in range(2):
    flash_attention_forward(Q[0, 0] / math.sqrt(D), K[0, 0], V[0, 0])
    attention_forward_triton(Q, K, V)
    attention_backward_triton(Q, K, V, dO)
torch.cuda.synchronize()

torch.cuda.cudart().cudaProfilerStart()
flash_attention_forward(Q[0, 0] / math.sqrt(D), K[0, 0], V[0, 0])
attention_forward_triton(Q, K, V)
attention_backward_triton(Q, K, V, dO)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
