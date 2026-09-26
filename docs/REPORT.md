# Evaluation Report: Triton Attention Kernels on an NVIDIA L4

**Hardware:** NVIDIA L4 (Ada, sm_89, 58 SMs, 24 GB GDDR6, ~300 GB/s, 48 MB L2, 100 KB shared memory per SM)
**Software:** PyTorch 2.11.0 (CUDA 13.0), Triton 3.6.0, driver 595.58, Nsight Compute 2024.3.2
**Precision:** float32 throughout; every `tl.dot` uses `input_precision="ieee"` (no TF32 tensor cores)

This report covers three implementations:

| Module | What it is |
|--------|-----------|
| `flash_attention/online_forward.py` | Fused FlashAttention-style forward: one program per query tile, online softmax over K/V tiles, N×N never stored. Single head `[N, D]`, unscaled. |
| `flash_attention/naive_triton.py` | Naive multi-head `[B, H, N, D]` forward + backward: 3 forward and 5 backward kernels, S / P / dP / dS materialised in global memory. |
| `flash_attention/reference.py` | PyTorch reference forward and a hand-derived backward, used as the oracle. |

---

## 1. Summary

1. **The fused online-softmax forward is correct** for every tested shape (N = 64…4096, including N not divisible by the tile size, D = 32/64/128) with max error ≤ 1.1e-5 against PyTorch.
2. **The original naive multi-head version failed for N ≥ 512 on the L4.** Its fix for the earlier `N = 256` bug set `BLOCK_N = next_power_of_2(N)`, so each program loaded all N keys into one tile. Shared memory need grows with N (135 KB at N = 512, 1 MB at N = 4096) and exceeds the L4's 99 KB per-block limit. It also failed at `D = 128` and at `N = 333`.
3. **Fix applied:** the score and dP kernels now tile the key axis over a third grid dimension (`BLOCK_N = 64`), and only the row-wise softmax kernels use a full-row block. After the fix all 47 tests pass, up to N = 4096.
4. **The naive backward is correct but slow:** 125 ms at N = 4096 versus 47 ms for PyTorch eager and 18 ms for PyTorch SDPA. Two thirds of that time (66 %) is spent in `backward_dk_kernel` and `backward_dv_kernel`, which read dS / P **column-wise** (strided, uncoalesced) and reduce with element-wise multiply-and-sum instead of `tl.dot`.
5. **Memory:** the naive path's activation memory grows as N² (2 GB for the backward at N = 4096, B·H = 8). The fused forward stays linear in N (24 MB).
6. **Nsight Compute could not collect hardware counters on this server** (see §6). Per-kernel times were collected with CUPTI tracing (`torch.profiler`) and compiled resource usage from Triton. Throughput and occupancy figures below are derived from those, not from `ncu`.

---

## 2. Correctness

All checks compare against `torch.softmax(QKᵀ·scale)·V` and, for gradients, PyTorch autograd of `sum(O ⊙ dO)`; tolerance `atol = rtol = 1e-4`.

### 2.1 Original code from the design discussion (unmodified)

| Case | Result |
|------|--------|
| Built-in tests `(B,H,N,D)`: (1,1,32,32), (1,2,64,32), (2,4,64,64), (2,4,128,64), (2,8,256,64) | pass; max abs error ≤ 7.2e-7 |
| N = 100, 200 | pass |
| N = 333, 512, 1024, 2048, 4096 (D = 64) | **fail**: `OutOfResources: shared memory` (135 KB → 1 MB required, 99 KB available) |
| D = 128 (N = 256) | **fail**: `OutOfResources: shared memory` (139 KB required) |
| D = 80 (not a power of two) | **fail**: compilation error (`tl.arange` needs a power-of-two size) |
| fp16 inputs | **fail**: compilation error (`tl.dot` of fp32 `P` with fp16 `V`) |
| Non-contiguous input (`[B,N,H,D].transpose(1, 2)`) | pass (all strides are passed explicitly) |

### 2.2 This repository (after the fix): `pytest`, 47 passed in 18.8 s

| Test file | Cases | Result |
|-----------|-------|--------|
| `test_reference.py` | manual backward vs autograd, float64, causal and non-causal, 3 shapes | pass (atol 1e-10) |
| `test_naive_triton.py` | forward + dQ/dK/dV for 12 shapes up to (1,1,4096,64), incl. N = 100, 333 and D = 128; unscaled mode | pass |
| `test_online_forward.py` | 9 (N, D, BLOCK_M, BLOCK_N) configs, N = 64…4096, D = 32/64/128 | pass |

### 2.3 Fused online-softmax forward

| N | D | BLOCK_M × BLOCK_N | max abs error |
|---|---|-------------------|---------------|
| 64 | 64 | 16 × 16 | 1.1e-5 |
| 128 | 64 | 32 × 32 | 6.4e-6 |
| 256 | 64 | 64 × 64 | 8.2e-6 |
| 1024 | 64 | 64 × 64 | 3.3e-6 |
| 4096 | 64 | 64 × 64 | 3.8e-6 |
| 100 | 64 | 32 × 32 | 8.5e-6 |
| 1000 | 64 | 64 × 64 | 2.4e-6 |
| 512 | 32 | 16 × 16 | 3.2e-6 |
| 256 | 128 | 64 × 64 | **OutOfResources** (180 KB shared memory needed); passes with 32 × 32 |

The debug stores in the original kernel (`temp_acc`, `temp_m`, `temp_l`, `temp_pid`) wrote to `tl.arange(0, BLOCK_M)` rather than `offs_m`, so every query tile overwrote the same slots. They were removed; the attention math is unchanged.

---

## 3. End-to-end latency and memory

`benchmarks/bench.py`, B = 1, H = 8, D = 64, float32, median of `triton.testing.do_bench`. Peak memory is the allocation above the inputs.

| N | Online fwd (8 heads, looped) | Naive fwd | Naive bwd | Eager fwd | Eager fwd+bwd | SDPA fwd | SDPA fwd+bwd |
|---|---|---|---|---|---|---|---|
| 256 | 0.36 ms / 1.5 MiB | 0.05 ms / 4.5 MiB | 0.35 ms / 9.5 MiB | 0.09 ms / 4.5 MiB | 0.20 ms / 9 MiB | 0.05 ms / 0.5 MiB | 0.17 ms / 3 MiB |
| 512 | 0.69 / 3 | 0.16 / 17 | 1.46 / 35 | 0.22 / 17 | 0.47 / 34 | 0.12 / 1 | 0.44 / 7 |
| 1024 | 1.33 / 6 | 0.70 / 66 | 7.10 / 134 | 0.92 / 66 | 2.84 / 132 | 0.39 / 2 | 1.34 / 12 |
| 2048 | 2.61 / 12 | 3.60 / 260 | 30.5 / 524 | 3.98 / 260 | 11.9 / 520 | 1.32 / 4 | 4.79 / 24 |
| 4096 | 8.01 / 24 | 15.0 / 1032 | 125.5 / 2072 | 15.1 / 1032 | 47.5 / 2064 | 5.03 / 8 | 18.1 / 48 |

Observations:

- **Memory:** naive Triton and PyTorch eager both store the N×N matrices, so memory grows 4× each time N doubles. The fused forward and SDPA grow linearly.
- **Naive forward** matches PyTorch eager (both are bound by writing and reading S and P).
- **Naive backward** is 2.6× slower than eager autograd at N = 4096 (see §4).
- **Online forward** is launched once per head from a Python loop (8 launches + a `torch.stack`). At small N this overhead dominates. At N = 4096 a single head takes 1.41 ms on the GPU (§4), about 11.3 ms for 8 heads, versus 5.0 ms for SDPA, which uses a tuned multi-head kernel.

---

## 4. Per-kernel analysis

GPU time is from CUPTI activity tracing (`benchmarks/kernel_stats.py`, mean of 10 iterations). Registers, spills, shared memory and warps come from the compiled Triton kernels. Achieved bandwidth counts **unique** bytes (each N×N fp32 tensor = 512 MiB at N = 4096, B·H = 8) against the L4's ~300 GB/s. Occupancy is **estimated** from registers and shared memory (65,536 registers, 100 KB shared memory, 48 warps per SM).

**N = 4096, B = 1, H = 8, D = 64** (online kernel: one head)

| Kernel | Time | TFLOP/s | Unique GB/s (% of DRAM peak) | Regs | Spills | Shared mem | Est. occupancy |
|--------|------|---------|-----------------------------|------|--------|-----------|----------------|
| `online_softmax_attention` (1 head) | 1.41 ms | 3.04 | — | 168 | 0 | 96 KB | 1 block/SM, 8 % |
| `forward_score_kernel` S = QKᵀ | 7.92 ms | 2.17 | 70 (23 %) | 128 | 0 | 20 KB | 4 blocks, 33 % |
| `forward_softmax_kernel` | 4.57 ms | — | 235 (78 %) | 51 | 0 | — | 75 % |
| `forward_output_kernel` O = PV | 2.56 ms | 6.71 | 216 (72 %) | 56 | 0 | 40 KB | 2 blocks, 17 % |
| `backward_dp_kernel` dP = dO Vᵀ | 7.79 ms | 2.21 | 71 (24 %) | 128 | 0 | 20 KB | 33 % |
| `backward_ds_kernel` softmax bwd | 6.93 ms | — | 232 (77 %) | 80 | 0 | — | 50 % |
| `backward_dq_kernel` dQ = dS K | 15.2 ms | 1.13 | 36 (12 %) | 39 | 0 | 1 KB | 100 % |
| `backward_dk_kernel` dK = dSᵀ Q | **42.2 ms** | 0.41 | 13 (4 %) | 40 | 0 | 1 KB | 100 % |
| `backward_dv_kernel` dV = Pᵀ dO | **41.0 ms** | 0.42 | 14 (5 %) | 40 | 0 | 1 KB | 100 % |

The backward launches the score and softmax kernels again to rebuild S and P, so one backward runs 7 kernels: 7.8 + 7.9 + 4.6 + 6.9 + 15.2 + 42.2 + 41.0 ≈ 125.6 ms, which matches the 125.5 ms end-to-end figure.

At N = 1024 the whole B·H·N² working set (32 MB) fits in the 48 MB L2 cache, so the element-wise kernels run faster than DRAM bandwidth would allow; N = 4096 is the representative case. The N = 1024 numbers are in `benchmarks/results/l4_kernel_stats.json`.

### 4.1 Findings

**`backward_dk_kernel` / `backward_dv_kernel`: 66 % of backward time.** Each program owns one key row `j` and walks down column `j` of dS (or P): `dS + offs_m * stride_dsi + pid_n * stride_dsj`. Consecutive threads read addresses `N × 4` bytes apart, so every 4-byte element needs its own 32-byte memory sector, which wastes 8× the bandwidth. The product is then formed as `tl.sum(ds[:, None] * q, axis=0)` on CUDA cores, not with `tl.dot`. They reach 4–5 % of DRAM bandwidth even at 100 % estimated occupancy, which points to memory access pattern rather than parallelism as the limiter.

**`backward_dq_kernel`: 12 % of peak.** It reads dS row-wise, which is coalesced, so it runs 2.8× faster than dK/dV. It still uses element-wise multiply-and-sum and re-reads all of K for every query row (N = 4096 times). That traffic is mostly absorbed by L2.

**Score / dP kernels: 23 % of peak.** These need 128 registers and 20 KB of shared memory at `BLOCK_M = 16`, which limits them to 4 blocks per SM. The 16 × 64 tiles are small, and `ieee` precision rules out tensor cores, so each program does little work per byte stored.

**Softmax, softmax-backward and PV kernels: 72–78 % of peak.** These are simple streaming kernels and are already close to the bandwidth limit. They are not worth optimising further in the naive design.

**Fused online-softmax forward.** 168 registers and 96 KB of shared memory (64 × 64 tiles, D = 64, fp32) allow only **one block per SM (4 warps, ~8 % occupancy)**. With a single head the grid has only 64 programs for 58 SMs: one full wave plus a 6-block tail. A `[B, H, N, D]` grid (B·H·N/BLOCK_M programs) would fill the GPU. Smaller tiles or fewer pipeline stages would allow more than one block per SM.

### 4.2 Recommended next steps (in order of payoff)

1. **Rewrite dK/dV as tile-level matmuls.** Let each program own a key tile, load `dS[m_tile, n_tile]` row-major (coalesced) and accumulate `tl.dot(tl.trans(ds), q)`. This removes the strided reads and moves the work to `tl.dot`. It is also the structure the real FlashAttention backward uses.
2. **Use `tl.dot` in dQ** in the same way.
3. **Extend the fused forward to `[B, H, N, D]`** with the `(query block, B·H)` grid and a `scale` argument, and save the row log-sum-exp `m_i + log(l_i)`.
4. **Implement the FlashAttention backward** using the saved log-sum-exp and `Δ = rowsum(dO ⊙ O)` to recompute P tile-by-tile, removing all four N×N buffers.
5. **Consider `input_precision="tf32"`** for `tl.dot` to use tensor cores. Expect errors around 1e-3 instead of 1e-6, so test tolerances would need adjusting.

---

## 5. Supported inputs and limits (this repository)

| Property | Naive Triton | Fused forward |
|----------|-------------|---------------|
| Layout | `[B, H, N, D]`, any strides | `[N, D]`, one head |
| Scale | `scale` argument (default `1/√D`) | none (fold into Q) |
| N | any; memory is O(B·H·N²) | any |
| D | power of two | power of two; D = 128 needs ≤ 32 × 32 tiles on L4 |
| dtype | float32 only | float32 |
| Causal mask | no | no |
| Softmax row block | `next_power_of_2(N)` in registers; very long rows (≳ 16k) will spill | n/a |

---

## 6. Nsight Compute status

`ncu` 2024.3.2 is installed on the test server, but profiling failed with:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access
NVIDIA GPU Performance Counters on the target device 0.
```

Cause: the host driver has `RmProfilingAdminOnly: 1`, and the workload runs inside a Docker container **without** `CAP_SYS_ADMIN`, so even `root` in the container is denied. This cannot be changed from inside the container. Either:

- run the container with `--cap-add SYS_ADMIN`, or
- set `options nvidia NVreg_RestrictProfilingToAdminUsers=0` on the host and reload the driver.

Once counters are available, `bash benchmarks/profile_ncu.sh` produces a full `.ncu-rep` (Speed of Light, occupancy, memory workload, source-level hot spots) for every kernel at N = 1024. The most useful metrics to confirm §4.1 are:

- **Memory coalescing in dK/dV:** `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` versus requests.
- **Achieved occupancy:** `sm__warps_active.avg.pct_of_peak_sustained_active`.
- **DRAM throughput:** `dram__throughput.avg.pct_of_peak_sustained_elapsed`.

---

## 7. Reproducing

```bash
pip install -r requirements.txt
pytest                                   # correctness (47 tests)
python benchmarks/bench.py --out bench.json
python benchmarks/kernel_stats.py --out stats.json
bash benchmarks/profile_ncu.sh           # needs GPU counter access
```

Raw results from this run: `benchmarks/results/l4_bench.json`, `benchmarks/results/l4_kernel_stats.json`.
