#!/usr/bin/env python3
"""Roofline chart: each kernel as a dot at (arithmetic intensity, achieved GFLOPS).

Arithmetic intensity (AI) = FLOPs per byte moved from global memory.
  * measured:    2N^3 / DRAM bytes, from results/ncu_metrics.csv (scripts/profile.sh)
  * theoretical: from the tile shape, ignoring caches. A block computing a BM x BN
                 tile loads (BM + BN) floats per k step and does 2*BM*BN FLOPs, so
                 AI = BM*BN / (2*(BM + BN)). Naive: BM = BN = 1 -> 0.25.
If the ncu file exists the measured values are used; GFLOPS come from bench.csv.

usage: python3 scripts/roofline.py [--n 4096] [--peak-gflops 8100 --peak-bw 320]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ncu_metrics
import style

# Approximate published FP32 peak (GFLOPS) and DRAM bandwidth (GB/s). Check your
# card's datasheet and override with --peak-gflops / --peak-bw.
KNOWN_GPUS = {
    "T4": (8100, 320),
    "P100": (9300, 732),
    "V100": (15700, 900),
    "A100": (19500, 1555),
    "A10G": (31200, 600),
    "L4": (30300, 300),
    "RTX 3060": (12700, 360),
    "RTX 3090": (35600, 936),
    "RTX 4070": (29100, 504),
    "RTX 4090": (82600, 1008),
}

TILES = {   # (BM, BN) per kernel, for the theoretical intensity
    "naive": (1, 1), "coalesced": (1, 1), "shared": (32, 32), "reg1d": (64, 64),
    "reg2d": (128, 128), "vectorized": (128, 128),
}
TUNED_TILES = [(128, 128), (128, 128), (128, 64), (64, 128), (64, 64), (64, 64), (128, 128)]
ID_TO_NAME = {0: "cublas", 1: "naive", 2: "coalesced", 3: "shared", 4: "reg1d", 5: "reg2d",
              6: "vectorized", 7: "tuned"}


def theoretical_ai(kernel, cfg):
    if kernel == "tuned":
        bm, bn = TUNED_TILES[int(cfg)]
    elif kernel in TILES:
        bm, bn = TILES[kernel]
    else:
        return None
    return bm * bn / (2.0 * (bm + bn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/bench.csv")
    ap.add_argument("--ncu", default="results/ncu_metrics.csv")
    ap.add_argument("--ncu-n", type=int, default=2048, help="N used by profile.sh")
    ap.add_argument("--n", type=int, default=4096, help="size whose GFLOPS are plotted")
    ap.add_argument("--peak-gflops", type=float)
    ap.add_argument("--peak-bw", type=float, help="GB/s")
    ap.add_argument("--out", default="results/roofline.png")
    a = ap.parse_args()
    style.apply()

    df = pd.read_csv(a.csv)
    df["kernel"] = df["kernel"].str.strip()
    d = df[(df["n"] == a.n) & (df["kernel"] != "cpu") & (df["pass"] != "NO")]
    if d.empty:
        raise SystemExit(f"no rows for n={a.n} in {a.csv}")
    gpu = d["gpu"].iloc[0]

    peak_gf, peak_bw = a.peak_gflops, a.peak_bw
    if peak_gf is None or peak_bw is None:
        match = next((v for k, v in KNOWN_GPUS.items() if k in gpu), None)
        if match is None:
            raise SystemExit(f"unknown GPU '{gpu}': pass --peak-gflops and --peak-bw")
        peak_gf = peak_gf or match[0]
        peak_bw = peak_bw or match[1]
    ridge = peak_gf / peak_bw

    measured = {}
    if os.path.exists(a.ncu):
        for kid, m in ncu_metrics.load(a.ncu).items():
            ai, _ = ncu_metrics.intensity(m, a.ncu_n)
            if ai == ai:
                measured[ID_TO_NAME.get(kid)] = ai
    source = "measured with Nsight Compute (DRAM bytes)" if measured else "theoretical, from tile sizes"

    fig, ax = plt.subplots(figsize=(9, 5.6))
    x = np.logspace(-2, 3, 400)
    ax.plot(x, np.minimum(peak_gf, x * peak_bw), color=style.NEUTRAL, linewidth=2)
    ax.axvline(ridge, color=style.GRID, linewidth=1, linestyle="--")
    ax.text(ridge * 1.08, peak_gf * 0.012, f"ridge point\n{ridge:.0f} FLOP/byte", color=style.TEXT_2, fontsize=8.5)
    ax.text(0.03, 0.03 * peak_bw * 1.5, f"memory roof {peak_bw:.0f} GB/s", color=style.TEXT_2, fontsize=8.5)
    ax.text(200, peak_gf * 1.12, f"compute roof {peak_gf / 1000:.1f} TFLOPS", color=style.TEXT_2,
            fontsize=8.5, ha="center")

    for k in style.KERNEL_ORDER:
        row = d[d["kernel"] == k]
        if row.empty:
            continue
        ai = measured.get(k) if measured else theoretical_ai(k, row["cfg"].iloc[0])
        if ai is None:
            continue   # cuBLAS has no theoretical value; it appears only when measured
        ax.scatter(ai, row["gflops"].iloc[0], s=90, color=style.KERNEL_COLORS[k],
                   edgecolor=style.SURFACE, linewidth=2, zorder=3, label=style.LABELS[k])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(peak_gf * 1e-3, peak_gf * 2)
    ax.set_xlabel("arithmetic intensity (FLOP / byte of global memory)")
    ax.set_ylabel("GFLOPS")
    ax.set_title(f"Roofline, {gpu}, N = {a.n}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), title="intensity: " + ("measured" if measured else "theoretical"))
    note = f"Intensity {source}. Peaks: {peak_gf:.0f} GFLOPS, {peak_bw:.0f} GB/s (approximate)."
    if not measured:
        note += "\nTheoretical intensity ignores L1/L2 caches, so a dot can sit above the memory roof."
    fig.text(0.01, -0.04, note, color=style.TEXT_2, fontsize=8.5)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
