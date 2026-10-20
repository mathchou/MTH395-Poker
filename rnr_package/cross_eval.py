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
    need = {(f"{ap.cell_str(m)}|{bank[m]['pstar']:g}", truth_specs(t)) for m in models for t in truths}
    need |= {(k, truth_specs(t)) for t in truths for k in ("cfr", "p0")}
    os.makedirs(OUT, exist_ok=True)
    cache_f = os.path.join(OUT, "evals_s0.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    todo = [p for p in need if f"{p[0]}||{p[1]}" not in cache]
    print(f"{len(models)} model cells x {len(truths)} true cells; {len(todo)} of {len(need)} evaluations to run "
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
    a = p.parse_args()
    if a.report:
        return report(a)
    if a.plot:
        return plot(a)
    run(a)
    report(a)


if __name__ == "__main__":
    main()
