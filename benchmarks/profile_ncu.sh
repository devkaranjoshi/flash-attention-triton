#!/usr/bin/env bash
# Nsight Compute profile of every kernel at B=1, H=8, N=1024, D=64.
#
# Needs access to GPU performance counters. If ncu reports ERR_NVGPUCTRPERM:
#   * bare metal / VM : load the driver with NVreg_RestrictProfilingToAdminUsers=0,
#                       or run ncu as root
#   * Docker          : start the container with --cap-add SYS_ADMIN
#
# Usage: bash benchmarks/profile_ncu.sh [output-name]
set -euo pipefail

OUT="${1:-attention_l4}"
cd "$(dirname "$0")/.."

ncu --profile-from-start off \
    --set full \
    --import-source yes \
    -o "$OUT" \
    python benchmarks/profile_target.py

# Text summary: time, throughput, occupancy, memory traffic per kernel
ncu --import "$OUT.ncu-rep" --page details \
    --section SpeedOfLight --section Occupancy --section MemoryWorkloadAnalysis \
    > "$OUT.txt"
echo "Wrote $OUT.ncu-rep and $OUT.txt"
