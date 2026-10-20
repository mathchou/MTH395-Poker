"""
Pessimistic re-evaluation of the independent-bias grid (results_het/).

The grid's worst cases use the exact punisher, whose choice among near-tied actions depends on
tie-breaking. As in reeval_pessimistic.py, the pessimistic punisher still maximises its own
payoff, but among actions within EPS chips of its best it picks the one worst for us.

Both sides of every comparison are re-scored: for each cell with restricted-response solves,
the CFR table, the p = 0 solve, and every restricted response, each with
    gain    EV against the biased pair, relative to the seat's equilibrium value
    worst   pessimistic worst case over both punishers, relative to equilibrium
written to results_het_pess/pess_<pair>_a<tag>_s<seat>.json. Resumable.

    python reeval_het.py --workers 8        # evaluate (about 1-2 h on 8 workers)
    python reeval_het.py --report           # safe-exploitation verdicts: exact vs pessimistic
"""

import argparse
import glob
import json
import os
import pickle
import re
import time
from multiprocessing import Pool

import numpy as np

HET, OUT = "results_het", "results_het_pess"
PAT = re.compile(r"rnr_([ORCF]{2})_a([0-9.]+(?:-[0-9.]+)?)_s(\d)_independent_selfish_p([0-9.]+)_strategy\.npy$")
TOL, MIN_GAIN = 0.05, 0.05
G = {}                                                    # per-worker state


def cells():
    """{(pair, tag, seat): {p: strategy path}} for every cell with at least one solve at p > 0."""
    out = {}
    for f in glob.glob(os.path.join(HET, "rnr_*_strategy.npy")):
        m = PAT.search(os.path.basename(f))               # suffixed re-solves do not match
        if m:
            out.setdefault((m.group(1), m.group(2), int(m.group(3))), {})[float(m.group(4))] = f
    return {k: v for k, v in out.items() if any(p > 0 for p in v)}


def alphas(tag):
    a = tag.split("-")
    return (float(a[0]), float(a[-1]))


def init(eps):
    from tree_engine import Tree
    from cfr_path import find_cfr
    from q1_alpha_sweep import perturb
    G["T"] = T = Tree()
    G["base"] = pickle.load(open(find_cfr(), "rb"))["table"]
    G["B"] = [T.from_table(G["base"], q) for q in range(3)]
    G["EQ"] = T.ev(G["B"])
    G["perturb"], G["eps"] = perturb, eps


def evaluate_cell(item):
    from reeval_pessimistic import respond_pess
    (pair, tag, s), strategies = item
    out = os.path.join(OUT, f"pess_{pair}_a{tag}_s{s}.json")
    if os.path.exists(out):
        return f"{pair} a{tag} s{s}: already done"
    t0 = time.time()
    T, base, U, e = G["T"], G["base"], G["T"].util, G["EQ"][s]
    opps = [o for o in range(3) if o != s]
    fix = {}
    for o, t, a in zip(opps, pair, alphas(tag)):
        fix[o] = G["B"][o] if (t == "O" or a == 0) else T.from_table(G["perturb"](base, t, a), o)

    def evaluate(strat):
        prof = [None] * 3; prof[s] = strat
        for o in opps:
            prof[o] = fix[o]
        worst = {}
        for o in opps:
            other = [x for x in opps if x != o][0]
            pr = [None] * 3; pr[s], pr[other] = strat, fix[other]
            worst[str(o)] = respond_pess(T, pr, o, U[:, o], U[:, s], G["eps"]) - e
        return {"gain": T.ev(prof)[s] - e, "worst": min(worst.values()), "worst_by_punisher": worst}

    res = {"pair": pair, "tag": tag, "seat": s, "eps": G["eps"], "cfr": evaluate(G["B"][s]),
           "rnr": {f"{p:g}": evaluate(np.load(f)) for p, f in sorted(strategies.items())}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    return f"{pair} a{tag} s{s}: {len(strategies)} strategies, {time.time() - t0:.0f}s"


def verdict(gb, wb, pts):
    """Best safe extra profit (0 if none) and largest safe p, from {p: (gain, worst)}."""
    safe = {p for p, (g, w) in pts.items() if w >= wb - TOL} | {0.0}
    wins = [g for p, (g, w) in pts.items() if p in safe and p > 0 and g >= gb + MIN_GAIN]
    return (max(wins) - gb if wins else 0.0), max(safe)


def report(grid=None):
    """grid: keep only cells whose alphas are multiples of this step (e.g. 0.2), so cells from
    partial runs on other grids are left out."""
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "pess_*.json"))):
        r = json.load(open(f)); pair, tag, s = r["pair"], r["tag"], r["seat"]
        if grid and not all(abs(x / grid - round(x / grid)) < 1e-9 for x in alphas(tag)):
            continue
        R = r["rnr"]
        if "0" not in R:
            continue
        # pessimistic
        gb = max(r["cfr"]["gain"], R["0"]["gain"]); wb = max(r["cfr"]["worst"], R["0"]["worst"])
        pess = verdict(gb, wb, {float(p): (v["gain"], v["worst"]) for p, v in R.items()})
        # exact, from the original results (same strategies)
        lin = json.load(open(os.path.join(HET, f"linear_{pair}_a{tag}_s{s}.json")))["rows"][0]
        ex = {}
        for p in R:
            j = json.load(open(os.path.join(HET, f"rnr_{pair}_a{tag}_s{s}_independent_selfish_p{float(p)}.json")))
            ex[float(p)] = j
        e = lin["vs_biased"] - r["cfr"]["gain"]           # equilibrium value implied by the CFR row
        egb = max(lin["vs_biased"], ex[0.0]["vs_biased"]) - e
        ewb = max(lin["worst_selfish"], ex[0.0]["worst_selfish"]) - e
        exact = verdict(egb, ewb, {p: (j["vs_biased"] - e, j["worst_selfish"] - e) for p, j in ex.items()})
        gain_check = max(abs((ex[float(p)]["vs_biased"] - e) - R[p]["gain"]) for p in R)
        rows.append(dict(cell=f"{pair} a{tag} s{s}", seat=s, exact=exact, pess=pess, gain_check=gain_check))
    if not rows:
        raise SystemExit(f"nothing in {OUT}/ yet")
    print(f"{len(rows)} cells; largest gain difference between the two evaluations (should be ~0): "
          f"{max(r['gain_check'] for r in rows):.1e}")
    for s in sorted({r["seat"] for r in rows}):
        R = [r for r in rows if r["seat"] == s]
        for name, k in (("exact", "exact"), ("pessimistic", "pess")):
            win = [r[k][0] for r in R if r[k][0] >= MIN_GAIN]
            print(f"  seat {s}, {name:<11}: safe exploitation in {len(win)} of {len(R)} cells; median extra "
                  f"{np.median(win) if win else 0:+.3f}; median largest safe p {np.median([r[k][1] for r in R]):.2f}")
    flips = [r for r in rows if (r["exact"][0] >= MIN_GAIN) != (r["pess"][0] >= MIN_GAIN)]
    print(f"cells whose verdict changes: {len(flips)}")
    for r in flips:
        print(f"  {r['cell']}: exact extra {r['exact'][0]:+.3f} -> pessimistic {r['pess'][0]:+.3f}")
    d = np.array([r["pess"][0] - r["exact"][0] for r in rows])
    print(f"change in extra profit, pessimistic minus exact: median {np.median(d):+.3f}, min {d.min():+.3f}, max {d.max():+.3f}")
    print("largest drops (pessimistic best safe point is a lower-profit one):")
    for r in sorted(rows, key=lambda r: r["pess"][0] - r["exact"][0])[:5]:
        print(f"  {r['cell']}: exact extra {r['exact'][0]:+.3f} (largest safe p {r['exact'][1]:g}) -> "
              f"pessimistic {r['pess'][0]:+.3f} (largest safe p {r['pess'][1]:g})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--grid", type=float, default=None, help="report only cells on this alpha grid, e.g. 0.2")
    a = ap.parse_args()
    if a.report:
        return report(a.grid)
    C = sorted(cells().items())
    todo = [c for c in C if not os.path.exists(os.path.join(OUT, f"pess_{c[0][0]}_a{c[0][1]}_s{c[0][2]}.json"))]
    print(f"{len(C)} cells with solves; {len(todo)} to evaluate on {a.workers} workers", flush=True)
    t0 = time.time()
    with Pool(a.workers, initializer=init, initargs=(a.eps,)) as pool:
        for i, msg in enumerate(pool.imap_unordered(evaluate_cell, todo), 1):
            print(f"[{(time.time() - t0) / 60:6.1f} min] {i}/{len(todo)}  {msg}", flush=True)


if __name__ == "__main__":
    main()
