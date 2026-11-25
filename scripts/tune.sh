#!/usr/bin/env bash
# Try every kernel-7 tile configuration and keep the fastest.
#   scripts/tune.sh          -> results/tune.csv, results/best_cfg.txt
#   N=2048 scripts/tune.sh   -> tune at another size
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./matmul
[[ -x $BIN ]] || { echo "build first: make" >&2; exit 1; }

N=${N:-4096}
OUT=results/tune.csv
mkdir -p results
$BIN --list | sed -n '/configs/,$p'
$BIN --csv-header > "$OUT"
NCFG=$($BIN --list | grep -cE '^  [0-9]  BM=')
for ((c = 0; c < NCFG; c++)); do
    $BIN --kernel 7 --cfg "$c" --n "$N" --reps 20 --csv | tee -a "$OUT"
done

# gflops is column 8, cfg column 4; only configs that passed the check count.
best=$(awk -F, 'NR > 1 && $13 == "yes" { if ($8 > g) { g = $8; c = $4 } } END { print c }' "$OUT")
echo "$best" > results/best_cfg.txt
echo "best config at N=$N: $best (saved to results/best_cfg.txt; bench.sh uses it)"
