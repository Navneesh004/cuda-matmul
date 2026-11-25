#!/usr/bin/env bash
# Profile every kernel with Nsight Compute.
#   scripts/profile.sh            -> results/ncu_metrics.csv (key metrics, all kernels)
#   FULL=1 scripts/profile.sh     -> also results/ncu/<kernel>.ncu-rep (open in the Nsight Compute GUI)
#
# The profiler replays each kernel many times, so profile at a smaller N than you
# benchmark. If ncu fails with ERR_NVGPUCTRPERM, this GPU blocks the hardware
# counters (common on cloud notebooks): use Kaggle or a local GPU instead.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./matmul
[[ -x $BIN ]] || { echo "build first: make" >&2; exit 1; }

N=${N:-2048}
KERNELS=${KERNELS:-"0 1 2 3 4 5 6 7"}
CFG=${CFG:-$(cat results/best_cfg.txt 2>/dev/null || echo 0)}
mkdir -p results/ncu

METRICS=(
    gpu__time_duration.sum                                      # kernel time
    dram__bytes_read.sum                                        # bytes from DRAM -> measured arithmetic intensity
    dram__bytes_write.sum
    dram__throughput.avg.pct_of_peak_sustained_elapsed          # memory throughput, % of peak
    sm__throughput.avg.pct_of_peak_sustained_elapsed            # compute throughput, % of peak
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum              # global load sectors ...
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum             # ... per request (4 = perfectly coalesced)
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum    # shared-memory bank conflicts (loads)
    l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum    # shared-memory bank conflicts (stores)
    sm__warps_active.avg.pct_of_peak_sustained_active           # achieved occupancy
    launch__registers_per_thread
)
METRIC_LIST=$(IFS=,; echo "${METRICS[*]}")

OUT=results/ncu_metrics.csv
: > "$OUT"
for k in $KERNELS; do
    # Our kernels are all named matmul_*; cuBLAS kernels contain "gemm".
    if [[ $k == 0 ]]; then filter="regex:gemm"; else filter="regex:matmul"; fi
    echo "profiling kernel $k at N=$N ..." >&2
    # --no-check: don't launch cuBLAS for the reference, so only our kernel is captured.
    ncu --csv --print-units base --metrics "$METRIC_LIST" -k "$filter" --launch-count 1 \
        "$BIN" --kernel "$k" --n "$N" --cfg "$CFG" --reps 1 --no-check \
        | awk -v k="$k" 'BEGIN{FS=OFS=","} /^"ID"/ { if (!hdr++) print "\"kernel_id\"", $0; next } /^"[0-9]/ { print "\"" k "\"", $0 }' \
        >> "$OUT.tmp"
    if [[ ${FULL:-0} == 1 ]]; then
        ncu --set full -k "$filter" --launch-count 1 -f -o "results/ncu/kernel$k" \
            "$BIN" --kernel "$k" --n "$N" --cfg "$CFG" --reps 1 --no-check > /dev/null
    fi
done
# Keep a single header line.
awk 'NR == 1 || !/^"kernel_id"/' "$OUT.tmp" > "$OUT" && rm -f "$OUT.tmp"
echo "wrote $OUT" >&2
python3 scripts/ncu_summary.py "$OUT" --n "$N" | tee results/ncu_summary.md
