#!/usr/bin/env bash
# Benchmark every kernel (plus cuBLAS and, for small N, the CPU) across sizes.
#   scripts/bench.sh                     -> results/bench.csv
#   SIZES="1024 4096" scripts/bench.sh   -> custom sizes
#   CFG=3 scripts/bench.sh               -> kernel 7 with config 3 (see scripts/tune.sh)
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./matmul
[[ -x $BIN ]] || { echo "build first: make" >&2; exit 1; }

SIZES=${SIZES:-"256 512 1024 2048 3072 4096 6144 8192"}
KERNELS=${KERNELS:-"0 1 2 3 4 5 6 7"}
CPU_MAX_N=${CPU_MAX_N:-1024}
REPS=${REPS:-10}
CFG=${CFG:-$(cat results/best_cfg.txt 2>/dev/null || echo 0)}
OUT=${OUT:-results/bench.csv}

mkdir -p results
$BIN --csv-header > "$OUT"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)   kernel 7 cfg: $CFG"

NAIVE_MAX_N=${NAIVE_MAX_N:-4096}   # kernels 1-2 take many seconds per run beyond this

for n in $SIZES; do
    if (( n <= CPU_MAX_N )); then
        $BIN --kernel cpu --n "$n" --reps 1 --csv | tee -a "$OUT"
    fi
    for k in $KERNELS; do
        reps=$REPS
        if [[ $k == 1 || $k == 2 ]]; then
            (( n > NAIVE_MAX_N )) && continue
            (( n >= 2048 )) && reps=3
        fi
        $BIN --kernel "$k" --n "$n" --reps "$reps" --cfg "$CFG" --csv | tee -a "$OUT"
    done
done
echo "wrote $OUT"
