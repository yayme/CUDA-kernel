"""Plot bench/results/results.csv: one figure per sparsity pattern, saved to assets/.

Run from the repo root:   python -m bench.plot
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bench.bench import RESULTS

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")


def main():
    # (window, stride) -> method -> list of (n, ms, gflops)
    data = defaultdict(lambda: defaultdict(list))
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            data[(int(r["window"]), int(r["stride"]))][r["method"]].append(
                (int(r["n"]), float(r["ms"]), float(r["useful_gflops"])))

    for (window, stride), methods in data.items():
        fig, (ax_ms, ax_tp) = plt.subplots(1, 2, figsize=(12, 4.5))
        for method, points in sorted(methods.items()):
            points.sort()
            ns = [p[0] for p in points]
            ax_ms.plot(ns, [p[1] for p in points], marker="o", label=method)
            ax_tp.plot(ns, [p[2] for p in points], marker="o", label=method)

        ax_ms.set(xscale="log", yscale="log", xlabel="sequence length N", ylabel="time (ms)", title="Runtime")
        ax_tp.set(xscale="log", yscale="log", xlabel="sequence length N", ylabel="useful GFLOP/s",
                  title="Throughput on active pairs only")
        for ax in (ax_ms, ax_tp):
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        fig.suptitle(f"window={window}, stride={stride}")
        fig.tight_layout()

        path = os.path.join(ASSETS, f"bench_w{window}_s{stride}.png")
        fig.savefig(path, dpi=120)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
