#!/usr/bin/env bash
# Everything, in order: build, test, tune, benchmark, profile, charts.
#   scripts/run_all.sh            (SKIP_PROFILE=1 to skip ncu, e.g. if counters are blocked)
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
make
bash scripts/test.sh
bash scripts/tune.sh
bash scripts/bench.sh
if [[ ${SKIP_PROFILE:-0} != 1 ]]; then
    bash scripts/profile.sh || echo "ncu failed (blocked counters?); roofline will use theoretical intensity"
fi
python3 scripts/plot.py
python3 scripts/roofline.py
echo "done: see results/"
