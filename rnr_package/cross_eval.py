"""
How wrong can the model be before the safe strategy stops being safe?

Every bank entry (a grid cell's restricted response at its largest safe p, p*) is played against
the TRUE opponents of every grid cell, not just its own. For each (model cell, true cell):
    margin   pessimistic worst case of the strategy against the true opponents, minus the best
             equilibrium-level strategy's (CFR or p = 0) against those same opponents
    gain     expected payoff against the true opponents, minus that baseline's
"Still safe" means margin >= -0.05, as in the paper. Evaluations only; no new solves.

Model errors are grouped by how far the true opponents are from the model:
    exact            same cell (the diagonal; should be safe by construction)
    alpha off        same types, alphas differ (by 0.2 on one opponent, 0.2 on both, 0.4+)
    one type wrong   one opponent's type differs (including equilibrium vs deviator)
    both wrong       both types differ

    python cross_eval.py --workers 8          # ~15 min for 63 x 63 cells (seat 0)
    python cross_eval.py --report
    python cross_eval.py --plot               # cross_eval/cross_eval_s0.png

The full bank matrix (every solved strategy at EVERY p, not just p*, against every cell's
opponents) and the question it answers -- is the true model's own restricted-response curve the
best a seat-0 player can do safely, or do strategies solved for other models do better?

    python cross_eval.py --all-p --workers 8  # every bank strategy x every cell (several hours)
    python cross_eval.py --envelope           # own curve vs the whole bank, for every cell
"""

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

import adapt_policies as ap

OUT = "cross_eval"
TOL = 0.05
G = {}


def init(pess):
    ap.init_sim(pess)
    G.update(ap.G)


def truth_specs(cell):
    t1, a1, t2, a2 = cell
    return f"{t1}{a1:g}-{t2}{a2:g}"


def evaluate(pair):
    return ap.evaluate(pair)


def run(a):
    init(a.pess)
    bank = G["bank"]
    models = sorted(c for c, b in bank.items() if b["pstar"] > 0 and b["pstar"] in b["paths"])[: a.cells]
    truths = sorted(bank)[: a.cells]
    if a.all_p:                                   # every solved strategy, not only each cell's p*
        keys = [f"{ap.cell_str(m)}|{p:g}" for m in sorted(bank)[: a.cells] for p in sorted(bank[m]["paths"]) if p > 0]
    else:
        keys = [f"{ap.cell_str(m)}|{bank[m]['pstar']:g}" for m in models]
    need = {(k, truth_specs(t)) for k in keys for t in truths}
    need |= {(k, truth_specs(t)) for t in truths for k in ("cfr", "p0")}
    os.makedirs(OUT, exist_ok=True)
    cache_f = os.path.join(OUT, "evals_s0.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    todo = [p for p in need if f"{p[0]}||{p[1]}" not in cache]
    print(f"{len(keys)} strategies x {len(truths)} true cells; {len(todo)} of {len(need)} evaluations to run "
          f"on {a.workers} workers", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.pess,)) as pool:
        for i, (pr, v) in enumerate(pool.imap_unordered(evaluate, todo), 1):
            cache[f"{pr[0]}||{pr[1]}"] = v
            if i % 100 == 0 or i == len(todo):
                json.dump(cache, open(cache_f, "w"))
                print(f"[{(time.time() - t0) / 60:6.1f} min] {i}/{len(todo)}", flush=True)
    json.dump(cache, open(cache_f, "w"))


def table(a):
    init(a.pess)
    bank = G["bank"]
    cache = json.load(open(os.path.join(OUT, "evals_s0.json")))
    models = sorted(c for c, b in bank.items() if b["pstar"] > 0 and b["pstar"] in b["paths"])
    truths = sorted(bank)
    E = lambda k, t: cache.get(f"{k}||{truth_specs(t)}")
    rows = []
    for m in models:
        k = f"{ap.cell_str(m)}|{bank[m]['pstar']:g}"
        for t in truths:
            v, c, z = E(k, t), E("cfr", t), E("p0", t)
            if not (v and c and z):
                continue
            wb, gb = max(c["worst"], z["worst"]), max(c["ev"], z["ev"])
            same_types = (m[0], m[2]) == (t[0], t[2])
            n_wrong = (m[0] != t[0]) + (m[2] != t[2])
            da = abs(m[1] - t[1]) + abs(m[3] - t[3])
            if m == t:
                cat = "exact"
            elif same_types:
                cat = "alpha off by 0.2" if da < 0.21 else ("alpha off, 0.4" if da < 0.41 else "alpha off, 0.6+")
            else:
                cat = "one type wrong" if n_wrong == 1 else "both types wrong"
            rows.append(dict(model=m, truth=t, cat=cat, margin=v["worst"] - wb, gain=v["ev"] - gb))
    return models, truths, rows


def report(a):
    _, _, rows = table(a)
    cats = ["exact", "alpha off by 0.2", "alpha off, 0.4", "alpha off, 0.6+", "one type wrong", "both types wrong"]
    print(f"{'true opponents vs model':<20}{'pairs':>7}{'still safe':>12}{'median margin':>15}"
          f"{'still profitable':>18}{'median gain':>13}")
    for c in cats:
        R = [r for r in rows if r["cat"] == c]
        if not R:
            continue
        safe = np.mean([r["margin"] >= -TOL for r in R]); prof = np.mean([r["gain"] >= 0.05 for r in R])
        print(f"{c:<20}{len(R):>7}{safe:>12.0%}{np.median([r['margin'] for r in R]):>+15.3f}"
              f"{prof:>18.0%}{np.median([r['gain'] for r in R]):>+13.3f}")
    print("\nstill safe: worst case within 0.05 of the best equilibrium-level strategy against the TRUE "
          "opponents (pessimistic punisher). still profitable: at least +0.05 over that strategy.")


def envelope(a):
    """For every true cell: the best SAFE expected payoff on the true model's own curve vs in the whole bank."""
    init(a.pess)
    bank = G["bank"]
    cache = json.load(open(os.path.join(OUT, "evals_s0.json")))
    rows = []
    for t in sorted(bank):
        ts = truth_specs(t)
        if f"cfr||{ts}" not in cache or f"p0||{ts}" not in cache:
            continue
        wb = max(cache[f"cfr||{ts}"]["worst"], cache[f"p0||{ts}"]["worst"])
        gb = max(cache[f"cfr||{ts}"]["ev"], cache[f"p0||{ts}"]["ev"])
        own, every, n = [0.0], [(0.0, "baseline")], 0
        for m in bank:
            for p in bank[m]["paths"]:
                if p == 0:
                    continue
                k = f"{ap.cell_str(m)}|{p:g}"; v = cache.get(f"{k}||{ts}")
                if v is None:
                    continue
                n += 1
                if v["worst"] >= wb - TOL:
                    every.append((v["ev"] - gb, k))
                    if m == t:
                        own.append(v["ev"] - gb)
        best = max(every)
        rows.append(dict(t=ts, own=max(own), bank=best[0], key=best[1], n=n,
                         same_types=best[1] != "baseline" and best[1].split("|")[0] != ts and
                         (best[1][0], best[1].split("-")[1][0]) == (ts[0], ts.split("-")[1][0])))
    if not rows:
        raise SystemExit("no evaluations yet: run with --all-p first")
    gap = np.array([r["bank"] - r["own"] for r in rows])
    print(f"{len(rows)} true cells, ~{int(np.median([r['n'] for r in rows]))} bank strategies scored against each")
    print(f"best safe profit on the true model's own curve:  median {np.median([r['own'] for r in rows]):+.3f}")
    print(f"best safe profit anywhere in the bank:           median {np.median([r['bank'] for r in rows]):+.3f}")
    print(f"gap (bank minus own): median {np.median(gap):+.3f}; above 0.05 in {np.mean(gap > 0.05):.0%} of cells; "
          f"above 0.20 in {np.mean(gap > 0.20):.0%}; own curve is (within 0.01) the best in {np.mean(gap < 0.01):.0%}")
    print(f"when another strategy wins, it was solved for the same types (different alphas) in "
          f"{np.mean([r['same_types'] for r in rows if r['bank'] - r['own'] > 0.05]):.0%} of cases")
    print("largest gaps:")
    for r in sorted(rows, key=lambda r: r["own"] - r["bank"])[:8]:
        print(f"  true {r['t']}: own curve {r['own']:+.3f}, bank {r['bank']:+.3f} ({r['key']})")


def plot(a):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    models, truths, rows = table(a)
    M = np.full((len(models), len(truths)), np.nan)
    for r in rows:
        M[models.index(r["model"]), truths.index(r["truth"])] = r["margin"]
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(np.clip(M, -0.6, 0.6), cmap="RdBu", vmin=-0.6, vmax=0.6, aspect="auto")
    lab = lambda c: f"{c[0]}{c[1]:g}/{c[2]}{c[3]:g}"
    ax.set_yticks(range(len(models)), [lab(m) for m in models], fontsize=6)
    ax.set_xticks(range(len(truths)), [lab(t) for t in truths], fontsize=6, rotation=90)
    ax.set_ylabel("model the strategy was solved for (at its largest safe p)")
    ax.set_xlabel("true opponents")
    fig.colorbar(im, ax=ax, shrink=0.6, label="worst case minus the equilibrium-level baseline's (red = less safe)")
    ax.set_title("Cross-evaluation, seat 0: safety when the model is wrong (diagonal = model correct)")
    plt.tight_layout(); out = os.path.join(OUT, "cross_eval_s0.png"); plt.savefig(out, dpi=110)
    print("plot:", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pess", action="store_true", help="p* from the pessimistic re-evaluation")
    p.add_argument("--report", action="store_true"); p.add_argument("--plot", action="store_true")
    p.add_argument("--cells", type=int, default=None, help="testing: only the first N cells")
    p.add_argument("--all-p", action="store_true", help="every solved strategy, not only p* (the bank matrix)")
    p.add_argument("--envelope", action="store_true", help="own curve vs the whole bank, for every cell")
    a = p.parse_args()
    if a.report:
        return report(a)
    if a.plot:
        return plot(a)
    if a.envelope:
        return envelope(a)
    run(a)
    envelope(a) if a.all_p else report(a)


if __name__ == "__main__":
    main()
