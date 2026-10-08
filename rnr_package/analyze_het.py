"""
Analyse the heterogeneous-opponent grid in results_het/ (overnight group H).

For each pair (e.g. FR from seat 0: seat 1 = F with bias alpha, seat 2 = R with bias alpha2)
and each (alpha, alpha2) cell, reports:
    BR gain            what a full best response earns over equilibrium (the potential)
    safe extra         extra profit of the best SAFE restricted response over the best
                       equilibrium-level strategy (CFR or RNR p = 0): at least +0.05 more
                       profit with a worst case no more than 0.05 below; '-' if none
    harder punisher    at p = 0.5, which opponent hurts us more by switching to a best
                       response: the first letter's seat, the second's, or '=' if within 0.05
All payoffs relative to our seat's equilibrium value; worst cases use the exact punisher.

    python analyze_het.py                   # tables, plus results_het/het_<pair>_s<seat>.png
"""

import glob
import json
import os
from collections import defaultdict

import numpy as np

HET = "results_het"


def seat_values():
    """Equilibrium value of each seat, from the OO baselines if present, else computed."""
    f = "results/linear_OO_a0.35_s{}.json"
    if all(os.path.exists(f.format(s)) for s in range(3)):
        return {s: json.load(open(f.format(s)))["rows"][0]["vs_biased"] for s in range(3)}
    import pickle
    from tree_engine import Tree
    from cfr_path import find_cfr
    T = Tree(); base = pickle.load(open(find_cfr(), "rb"))["table"]
    ev = T.ev([T.from_table(base, q) for q in range(3)])
    return {s: ev[s] for s in range(3)}


def load():
    lin, rnr = {}, defaultdict(dict)
    for f in glob.glob(f"{HET}/linear_*.json"):
        r = json.load(open(f)); lin[(r["pair"], r["seat"], r["alpha"], r["alpha2"])] = r["rows"]
    for f in glob.glob(f"{HET}/rnr_*_independent_selfish_p*.json"):
        r = json.load(open(f))
        if "alpha2" in r:
            rnr[(r["pair"], r["seat"], r["alpha"], r["alpha2"])][r["p"]] = r
    return lin, rnr


def cell(lin_rows, pts, e, opps):
    if lin_rows is None or 0.0 not in pts:
        return None
    G = lambda r: r["vs_biased"] - e
    W = lambda r: r["worst_selfish"] - e
    c, b, p0 = lin_rows[0], lin_rows[-1], pts[0.0]
    gb, wb = max(G(c), G(p0)), max(W(c), W(p0))
    safe = [(G(r), W(r), p) for p, r in pts.items() if p > 0 and G(r) - gb >= 0.05 and W(r) >= wb - 0.05]
    best = max(safe) if safe else None
    harder = None
    if 0.5 in pts:
        r = pts[0.5]; v = [r[f"selfish_seat{o}"] for o in opps]
        harder = "=" if abs(v[0] - v[1]) < 0.05 else (0 if v[0] < v[1] else 1)
    return dict(br=G(b), extra=(best[0] - gb) if best else None, best=best, harder=harder,
                n_rnr=sum(1 for p in pts if p > 0))


def main():
    EQ = seat_values()
    lin, rnr = load()
    keys = sorted(set(lin) | set(rnr))
    if not keys:
        raise SystemExit(f"no results in {HET}/")
    for pair, seat in sorted({(k[0], k[1]) for k in keys}):
        opps = [o for o in range(3) if o != seat]
        A = sorted({k[2] for k in keys if k[:2] == (pair, seat)})
        B = sorted({k[3] for k in keys if k[:2] == (pair, seat)})
        C = {(a, b): cell(lin.get((pair, seat, a, b)), rnr.get((pair, seat, a, b), {}), EQ[seat], opps)
             for a in A for b in B}
        name = {0: f"seat {opps[0]} ({pair[0]})", 1: f"seat {opps[1]} ({pair[1]})", "=": "similar"}
        print(f"\n=== {pair}, we sit in seat {seat}: rows = seat {opps[0]}'s ({pair[0]}) alpha, "
              f"columns = seat {opps[1]}'s ({pair[1]}) alpha")
        for title, fn in (("BR gain over equilibrium", lambda c: f"{c['br']:+.3f}"),
                          ("safe extra profit (best safe RNR minus best equilibrium-level)",
                           lambda c: f"{c['extra']:+.3f}" if c["extra"] is not None else "   -  "),
                          ("harder punisher at p = 0.5", lambda c: {0: f"   {pair[0]}  ", 1: f"   {pair[1]}  ", "=": "   =  ",
                                                                   None: "   ?  "}[c["harder"]])):
            print(f"\n  {title}")
            print("  " + " " * 8 + "".join(f"{b:>9}" for b in B))
            for a in A:
                print(f"  {a:>8}" + "".join(f"{(fn(C[(a, b)]) if C[(a, b)] else 'n/a'):>9}" for b in B))
        n = sum(1 for c in C.values() if c); ns = sum(1 for c in C.values() if c and c["n_rnr"] < 3)
        print(f"\n  cells with results: {n} of {len(C)}" + (f"; {ns} still missing some p values" if ns else ""))
        try:
            plot(pair, seat, opps, A, B, C)
        except Exception as ex:     # plotting is optional
            print("  (plot skipped:", ex, ")")


def plot(pair, seat, opps, A, B, C):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = (("BR gain over equilibrium", lambda c: c["br"], "viridis"),
              ("safe extra profit\n(blank = no safe exploitation found)", lambda c: c["extra"], "viridis"),
              (f"harder punisher at p = 0.5\n(1 = seat {opps[0]} {pair[0]}, 2 = seat {opps[1]} {pair[1]}, 0 = similar)",
               lambda c: {0: 1, 1: 2, "=": 0}.get(c["harder"]), "coolwarm"))
    for ax, (title, fn, cmap) in zip(axs, panels):
        M = np.full((len(A), len(B)), np.nan)
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                c = C[(a, b)]
                v = fn(c) if c else None
                if v is not None: M[i, j] = v
        im = ax.imshow(M, cmap=cmap, origin="lower")
        for i in range(len(A)):
            for j in range(len(B)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}" if "punisher" not in title else f"{int(M[i, j])}",
                            ha="center", va="center", fontsize=9, color="white")
        ax.set_xticks(range(len(B)), [f"{b:g}" for b in B]); ax.set_yticks(range(len(A)), [f"{a:g}" for a in A])
        ax.set_xlabel(f"seat {opps[1]} ({pair[1]}) alpha"); ax.set_ylabel(f"seat {opps[0]} ({pair[0]}) alpha")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{pair}, we sit in seat {seat}: heterogeneous biases (exact punisher)", fontsize=12)
    plt.tight_layout()
    out = f"{HET}/het_{pair}_s{seat}.png"; plt.savefig(out, dpi=110); plt.close()
    print(f"  plot: {out}")


if __name__ == "__main__":
    main()
