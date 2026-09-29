"""
Summarise results/ into tables and frontier plots. Send me the printed output (or the
results/ folder) and I will fold it into the paper.

    python summarize.py
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

groups = defaultdict(list)
for f in glob.glob("results/rnr_*.json"):
    r = json.load(open(f))
    groups[(r["pair"], r["alpha"], r["seat"])].append(r)

WARN = 0.005
for key in sorted(groups):
    pair, a, seat = key
    print(f"\n=== {pair}  alpha {a}  seat {seat}")
    lf = f"results/linear_{pair}_a{a}_s{seat}.json"
    if os.path.exists(lf):
        print("  linear mixing:")
        for row in json.load(open(lf))["rows"]:
            print(f"    w={row['w']:<4} vs_biased {row['vs_biased']:+.3f}  "
                  f"worst_selfish {row['worst_selfish']:+.3f}  worst_adv {row['worst_adversary']:+.3f}")
    for restrict, free in sorted({(r["restrict"], r["free"]) for r in groups[key]}):
        rows = sorted([r for r in groups[key] if r["restrict"] == restrict and r["free"] == free],
                      key=lambda r: r["p"])
        print(f"  RNR {restrict} / {free}:")
        for r in rows:
            g = max(r["gaps"].values())
            flag = "  <-- not converged, rerun with more --iters" if g > WARN else ""
            print(f"    p={r['p']:<5} vs_biased {r['vs_biased']:+.3f}  worst_selfish "
                  f"{r['worst_selfish']:+.3f}  worst_adv {r['worst_adversary']:+.3f}  "
                  f"max gap {g:.4f}{flag}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    if os.path.exists(lf):
        rows = json.load(open(lf))["rows"]
        ax.plot([r["worst_selfish"] for r in rows], [r["vs_biased"] for r in rows], "-o",
                color="gray", label="linear mixing")
    for restrict, free in sorted({(r["restrict"], r["free"]) for r in groups[key]}):
        rows = sorted([r for r in groups[key] if r["restrict"] == restrict and r["free"] == free],
                      key=lambda r: r["p"])
        ax.plot([r["worst_selfish"] for r in rows], [r["vs_biased"] for r in rows], "-s",
                label=f"RNR {restrict}/{free}")
    ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel("worst case over both opponents (self-interested punisher)")
    ax.set_ylabel(f"EV vs biased pair {pair}")
    ax.set_title(f"{pair}, alpha {a}, seat {seat}")
    ax.grid(alpha=.3); ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"results/frontier_{pair}_a{a}_s{seat}.png", dpi=120); plt.close()
print("\nplots written to results/frontier_*.png")
