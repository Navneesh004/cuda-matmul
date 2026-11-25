#!/usr/bin/env python3
"""Markdown table of the key Nsight Compute metrics per kernel.

usage: python3 scripts/ncu_summary.py results/ncu_metrics.csv --n 2048
"""
import argparse

import ncu_metrics

NAMES = {0: "cuBLAS", 1: "1 naive", 2: "2 coalesced", 3: "3 shared", 4: "4 reg1d",
         5: "5 reg2d", 6: "6 vectorised", 7: "7 tuned"}


def fmt(x, spec):
    return "n/a" if x != x else format(x, spec)   # x != x is NaN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    data = ncu_metrics.load(a.csv)

    print(f"Nsight Compute, N = {a.n} (one launch per kernel)\n")
    print("| kernel | time (ms) | GFLOPS | DRAM bytes | FLOP/byte | DRAM % peak | SM % peak "
          "| sectors/request (ld) | bank conflicts ld / st | occupancy % | regs |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(data):
        m = data[k]
        nan = float("nan")
        ai, gf = ncu_metrics.intensity(m, a.n)
        sect = m.get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", nan)
        req = m.get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", nan)
        spr = sect / req if req and req == req else nan
        dram = m.get("dram__bytes_read.sum", 0) + m.get("dram__bytes_write.sum", 0)
        print(f"| {NAMES.get(k, k)} "
              f"| {fmt(m.get('gpu__time_duration.sum', nan) * 1e3, '.3f')} "
              f"| {fmt(gf, ',.0f')} "
              f"| {dram / 1e6:,.1f} MB "
              f"| {fmt(ai, '.1f')} "
              f"| {fmt(m.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', nan), '.1f')} "
              f"| {fmt(m.get('sm__throughput.avg.pct_of_peak_sustained_elapsed', nan), '.1f')} "
              f"| {fmt(spr, '.2f')} "
              f"| {fmt(m.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum', nan), ',.0f')} / "
              f"{fmt(m.get('l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum', nan), ',.0f')} "
              f"| {fmt(m.get('sm__warps_active.avg.pct_of_peak_sustained_active', nan), '.1f')} "
              f"| {fmt(m.get('launch__registers_per_thread', nan), '.0f')} |")
    print("\nsectors/request: 4 means a warp's 32 floats came in four 32-byte sectors (perfectly "
          "coalesced); up to 32 means every thread needed its own sector.")


if __name__ == "__main__":
    main()
