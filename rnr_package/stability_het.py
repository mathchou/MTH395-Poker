"""
Stability of the independent-bias grid's best safe points.

The best safe restricted responses sit close to the abrupt switch from safe to exposed, where
solutions may be least stable. This re-solves a sample of cells at exactly their best safe p:
    seed1     same 600 iterations, random first-iteration strategies   (all sampled cells)
    it1200    twice the iterations                                      (half of them)
and compares profit, worst case and the safe/unsafe verdict with the original solve.

Cells are chosen from those with safe exploitation, spread evenly over each seat's list
ordered by best safe p (so the high-p cells nearest the switch are well represented).
Re-solves go to results_het_check/, never results_het/, so they cannot mix into the grid.

    python stability_het.py --list            # the sample and its jobs
    python stability_het.py                   # run (about 5 h on 8 workers)
    python stability_het.py --report          # compare with the originals
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HET, CHECK = "results_het", "results_het_check"
TOL, MIN_GAIN = 0.05, 0.05
PY = sys.executable


def best_points():
    """[(seat, pair, tag, best p, baseline gain, baseline worst, eq value)] for cells with safe exploitation."""
    pat = re.compile(r"rnr_([ORCF]{2})_a([0-9.]+(?:-[0-9.]+)?)_s(\d)_independent_selfish_p([0-9.]+)\.json$")
    R = {}
    for f in glob.glob(os.path.join(HET, "rnr_*_independent_selfish_p*.json")):
        m = pat.search(os.path.basename(f))
        if m:
            R.setdefault((int(m.group(3)), m.group(1), m.group(2)), {})[float(m.group(4))] = json.load(open(f))
    eq = {s: json.load(open(os.path.join(HET, f"linear_OO_a0.0_s{s}.json")))["rows"][0]["vs_biased"]
          for s in range(3) if os.path.exists(os.path.join(HET, f"linear_OO_a0.0_s{s}.json"))}
    out = []
    for (s, pair, tag), pts in R.items():
        lin = os.path.join(HET, f"linear_{pair}_a{tag}_s{s}.json")
        if 0.0 not in pts or s not in eq or not os.path.exists(lin) or len(pts) < 3:
            continue
        L = json.load(open(lin))["rows"][0]; e = eq[s]
        gb = max(L["vs_biased"], pts[0.0]["vs_biased"]) - e
        wb = max(L["worst_selfish"], pts[0.0]["worst_selfish"]) - e
        wins = [(r["vs_biased"] - e, p) for p, r in pts.items()
                if p > 0 and r["worst_selfish"] - e >= wb - TOL and r["vs_biased"] - e >= gb + MIN_GAIN]
        if wins:
            out.append((s, pair, tag, max(wins)[1], gb, wb, e))
    return out


def sample(n):
    """n cells spread over each seat's list (ordered by best safe p), interleaved across seats with
    the highest p first, so the doubled-iteration half covers every seat and favours the switch."""
    pts = best_points()
    per = {s: sorted([x for x in pts if x[0] == s], key=lambda x: (x[3], x[1], x[2])) for s in range(3)}
    k = {0: n - 2 * (n // 3), 1: n // 3, 2: n // 3}
    picked = {}
    for s, L in per.items():
        idx = np.unique(np.linspace(0, len(L) - 1, min(k[s], len(L))).round().astype(int)) if L else []
        picked[s] = [L[i] for i in idx][::-1]
    out = []
    for i in range(max((len(v) for v in picked.values()), default=0)):
        for s in range(3):
            if i < len(picked[s]):
                out.append(picked[s][i])
    return out


def jobs(cells, iters):
    J = []
    for i, (s, pair, tag, p, *_rest) in enumerate(cells):
        a = tag.split("-"); a1, a2 = a[0], a[-1]
        base = ["rnr_general.py", "--pair", pair, "--alpha", a1, "--alpha2", a2, "--seat", str(s),
                "--p", str(p), "--restrict", "independent", "--free", "selfish", "--out", CHECK]
        variants = [("seed1", ["--iters", str(iters), "--seed", "1"])]
        if i < len(cells) // 2:
            variants.append((f"it{2 * iters}", ["--iters", str(2 * iters)]))
        for suf, extra in variants:
            out = os.path.join(CHECK, f"rnr_{pair}_a{tag}_s{s}_independent_selfish_p{p}_{suf}.json")
            J.append((f"{pair}_a{tag}_s{s}_p{p}_{suf}", base + extra + ["--suffix", suf], out))
    return J


def report(cells, iters):
    rows = []
    for (s, pair, tag, p, gb, wb, e) in cells:
        orig = json.load(open(os.path.join(HET, f"rnr_{pair}_a{tag}_s{s}_independent_selfish_p{p}.json")))
        for suf in ("seed1", f"it{2 * iters}"):
            f = os.path.join(CHECK, f"rnr_{pair}_a{tag}_s{s}_independent_selfish_p{p}_{suf}.json")
            if not os.path.exists(f):
                continue
            new = json.load(open(f))
            dg, dw = new["vs_biased"] - orig["vs_biased"], new["worst_selfish"] - orig["worst_selfish"]
            still = new["worst_selfish"] - e >= wb - TOL and new["vs_biased"] - e >= gb + MIN_GAIN
            rows.append((f"{pair} a{tag} s{s} p{p}", suf, dg, dw, still))
    if not rows:
        raise SystemExit("no re-solves yet")
    print(f"{'cell':<26}{'re-solve':<9}{'profit change':>14}{'worst change':>14}  still safe exploitation?")
    for c, suf, dg, dw, ok in rows:
        print(f"{c:<26}{suf:<9}{dg:>+14.3f}{dw:>+14.3f}  {'yes' if ok else 'NO'}")
    dg = np.array([r[2] for r in rows]); dw = np.array([r[3] for r in rows])
    print(f"\n{len(rows)} re-solves: profit change median {np.median(np.abs(dg)):.3f} (max {np.abs(dg).max():.3f}); "
          f"worst-case change median {np.median(np.abs(dw)):.3f} (max {np.abs(dw).max():.3f}); "
          f"verdict kept in {sum(r[4] for r in rows)} of {len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=16)
    ap.add_argument("--iters", type=int, default=600, help="iterations of the original solves")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    cells = sample(a.cells)
    if a.report:
        return report(cells, a.iters)
    J = jobs(cells, a.iters)
    if a.list:
        for s, pair, tag, p, gb, wb, e in cells:
            print(f"seat {s}  {pair} a{tag}  best safe p {p}")
        print(f"{len(J)} re-solves, {sum(not os.path.exists(o) for _, _, o in J)} still to run")
        return
    os.makedirs(CHECK, exist_ok=True); os.makedirs(os.path.join(CHECK, "logs"), exist_ok=True)
    queue = [j for j in J if not os.path.exists(j[2])]
    print(f"{len(queue)} re-solves on {a.workers} workers", flush=True)
    running, t0 = {}, time.time()
    try:
        while queue or running:
            while queue and len(running) < a.workers:
                name, argv, out = queue.pop(0)
                log = open(os.path.join(CHECK, "logs", name + ".log"), "w")
                running[name] = (subprocess.Popen([PY, "-u"] + argv, stdout=log, stderr=subprocess.STDOUT), log)
            time.sleep(5)
            for name in list(running):
                proc, log = running[name]
                if proc.poll() is not None:
                    log.close(); del running[name]
                    print(f"[{(time.time() - t0) / 60:6.1f} min] {'done' if proc.returncode == 0 else 'FAILED'} {name}",
                          flush=True)
    except KeyboardInterrupt:
        for proc, log in running.values():
            proc.terminate(); log.close()
        print("\nstopped; rerun to resume")


if __name__ == "__main__":
    main()
