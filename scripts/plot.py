#!/usr/bin/env python3
"""Charts from results/bench.csv.

  results/timing.png   GFLOPS per kernel at one size, cuBLAS as the last bar
  results/sizes.png    GFLOPS vs matrix size for every kernel
  results/summary.md   the same numbers as a Markdown table (for the README)

usage: python3 scripts/plot.py [--csv results/bench.csv] [--n 4096]
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

import style


def load(path):
    df = pd.read_csv(path)
    df["kernel"] = df["kernel"].str.strip()
    bad = df[(df["pass"] == "NO")]
    if len(bad):
        print("WARNING: these runs FAILED the correctness check and are excluded:")
        print(bad[["kernel", "n", "rel_err"]].to_string(index=False))
        df = df[df["pass"] != "NO"]
    return df


def timing_chart(df, n, out):
    d = df[(df["n"] == n) & (df["kernel"] != "cpu")].copy()
    if d.empty:
        raise SystemExit(f"no rows for n={n}; pick one of {sorted(df['n'].unique())}")
    d["order"] = d["kernel"].map(style.KERNEL_ORDER.index)
    d = d.sort_values("order")
    cublas = d.loc[d["kernel"] == "cublas", "gflops"]
    cublas = float(cublas.iloc[0]) if len(cublas) else None

    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = [style.LABELS[k] for k in d["kernel"]]
    # One series, so one hue; cuBLAS (the reference) in neutral grey.
    colors = [style.NEUTRAL if k == "cublas" else style.KERNEL_COLORS["naive"] for k in d["kernel"]]
    bars = ax.bar(labels, d["gflops"], color=colors, width=0.6, edgecolor=style.SURFACE, linewidth=2)
    top = d["gflops"].max()
    for bar, g in zip(bars, d["gflops"]):
        txt = f"{g:,.0f}"
        if cublas:
            txt += f"\n{100 * g / cublas:.1f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.01, txt,
                ha="center", va="bottom", fontsize=9, color=style.TEXT_2)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("GFLOPS (higher is better)")
    ax.grid(axis="x", visible=False)
    gpu = d["gpu"].iloc[0]
    ax.set_title(f"SGEMM throughput, N = {n} on {gpu}  (labels: GFLOPS, % of cuBLAS)")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def sizes_chart(df, out):
    d = df[df["kernel"] != "cpu"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for k in style.KERNEL_ORDER:
        s = d[d["kernel"] == k].sort_values("n")
        if s.empty:
            continue
        c = style.KERNEL_COLORS[k]
        ax.plot(s["n"], s["gflops"], color=c, linewidth=2, marker="o", markersize=5,
                markeredgecolor=style.SURFACE, markeredgewidth=1.5, label=style.LABELS[k])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ns = sorted(d["n"].unique())
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.set_xlabel("matrix size N")
    ax.set_ylabel("GFLOPS (log scale)")
    ax.set_title(f"Throughput vs matrix size on {d['gpu'].iloc[0]}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def summary_table(df, n, out):
    d = df[df["n"] == n].copy()
    d["order"] = d["kernel"].map(lambda k: -1 if k == "cpu" else style.KERNEL_ORDER.index(k))
    d = d.sort_values("order")
    cublas = d.loc[d["kernel"] == "cublas", "gflops"]
    cublas = float(cublas.iloc[0]) if len(cublas) else float("nan")
    lines = [f"GPU: **{d['gpu'].iloc[0]}**, N = {n}\n",
             "| kernel | time (ms) | GFLOPS | % of cuBLAS | speed-up vs previous |",
             "|---|---:|---:|---:|---:|"]
    prev = None
    for _, r in d.iterrows():
        sp = f"{r['gflops'] / prev:.2f}x" if prev and r["kernel"] not in ("cpu", "cublas") else ""
        lines.append(f"| {style.LABELS[r['kernel']]} | {r['ms_median']:.3f} | {r['gflops']:,.1f} | "
                     f"{100 * r['gflops'] / cublas:.1f}% | {sp} |")
        if r["kernel"] not in ("cpu", "cublas"):
            prev = r["gflops"]
    h2d = d["h2d_ms"].iloc[0]
    lines.append(f"\nHost-to-device copy of A and B: {h2d:.2f} ms (not included in kernel times).")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/bench.csv")
    ap.add_argument("--n", type=int, default=4096, help="size used for the bar chart and table")
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    style.apply()
    df = load(a.csv)
    os.makedirs(a.outdir, exist_ok=True)
    timing_chart(df, a.n, os.path.join(a.outdir, "timing.png"))
    sizes_chart(df, os.path.join(a.outdir, "sizes.png"))
    summary_table(df, a.n, os.path.join(a.outdir, "summary.md"))


if __name__ == "__main__":
    main()
