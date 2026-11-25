"""Parse the long-format CSV written by `ncu --csv` (via scripts/profile.sh)."""
import csv

TIME_SCALE = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0}
BYTE_SCALE = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
              "KB": 1e3, "MB": 1e6, "GB": 1e9}


def _num(s):
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load(path):
    """Return {kernel_id: {metric_name: value_in_base_units}} (first launch per kernel)."""
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            k = int(row["kernel_id"])
            name, unit, val = row["Metric Name"], row.get("Metric Unit", ""), _num(row["Metric Value"])
            if unit in TIME_SCALE:
                val *= TIME_SCALE[unit]       # -> seconds
            elif unit in BYTE_SCALE:
                val *= BYTE_SCALE[unit]       # -> bytes
            m = out.setdefault(k, {})
            # A kernel id can have several launches (e.g. cuBLAS helpers): sum time/bytes, keep the first otherwise.
            if name in m and (unit in TIME_SCALE or unit in BYTE_SCALE):
                m[name] += val
            else:
                m.setdefault(name, val)
    return out


def intensity(m, n):
    """Measured arithmetic intensity (FLOP per DRAM byte) and achieved GFLOPS."""
    flops = 2.0 * n ** 3
    dram = m.get("dram__bytes_read.sum", 0) + m.get("dram__bytes_write.sum", 0)
    t = m.get("gpu__time_duration.sum", float("nan"))
    ai = flops / dram if dram > 0 else float("nan")
    return ai, flops / t / 1e9
