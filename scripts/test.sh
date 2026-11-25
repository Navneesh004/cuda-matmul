#!/usr/bin/env bash
# Correctness test: every GPU kernel against cuBLAS at sizes that break naive
# assumptions (not multiples of the tile size, not multiples of 4, tiny).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./matmul
[[ -x $BIN ]] || { echo "build first: make" >&2; exit 1; }

SIZES=(1 7 31 33 64 100 127 128 129 255 1000 1023 1024 1025 2048)
KERNELS=(1 2 3 4 5 6 7)
NCFG=$($BIN --list | grep -cE '^  [0-9]  BM=')

fails=0; total=0
check() {   # args: kernel n [extra...]
    total=$((total + 1))
    if out=$($BIN --kernel "$1" --n "$2" --reps 1 "${@:3}" --csv); then
        :
    else
        fails=$((fails + 1)); echo "FAIL kernel=$1 n=$2 ${*:3}: $out"
    fi
}

for n in "${SIZES[@]}"; do
    for k in "${KERNELS[@]}"; do check "$k" "$n"; done
    for ((c = 1; c < NCFG; c++)); do check 7 "$n" --cfg "$c"; done
    printf '.'
done
echo
echo "$((total - fails))/$total passed"
[[ $fails -eq 0 ]]
