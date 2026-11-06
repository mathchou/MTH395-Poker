"""
How fast can the identifier forget and still classify steady opponents correctly?

With forgetting factor lam the identifier remembers about 1/(1 - lam) hands. Shorter memory catches a
switch sooner, but leaves less evidence for telling similar tables apart. This sweeps lam and measures,
with seat 0 playing p = 0 throughout (so identification alone is tested):

  steady accuracy   share of checkpoints, from hand 100 until any switch, at which the most likely
                    table (grid cell) is the true one AND confidence exceeds tau
  detection delay   hands after a switch until that first happens for the new table
                    (sessions that never get there count as the hands remaining)

The steady part uses the steady scenarios on the grid and the pre-switch phase of every switching
scenario; the off-grid steady scenario is left out (its true table lies between cells).

    python memory_sweep.py                     # ~10 min on 8 workers
    python memory_sweep.py --report            # tables from saved sessions
"""

import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import adapt_policies as ap

OUT = "memsweep"
LAMS = [1.0, 0.99, 0.98, 0.97, 0.95, 0.93, 0.9]
NEVER = "step1.01"                     # never commits: plays p = 0, but logs the posterior's best table


def sim_file(scen, seed, lam):
    return os.path.join(OUT, f"sim_{scen}_s{seed}_f{lam:g}.json")


def run(job):
    scen, seed, lam, hands, every = job
    return job, ap.simulate((scen, seed, NEVER, hands, every, lam, 0.0))


def truth_cell(scen, hand):
    s1, s2 = ap.SCENARIOS[scen]
    return ap.cell_str(ap.nearest_cell_label(*ap.style_at(s1, hand)) + ap.nearest_cell_label(*ap.style_at(s2, hand)))


def switch_hand(scen):
    s = {h for spec in ap.SCENARIOS[scen] for h, _, _ in ap.phases(spec)} - {0}
    return min(s) if s else None


def measure(r, tau, start=100):
    """(steady accuracy or None, detection delay or None) for one session."""
    scen, hands = r["scenario"], r["hands"]
    sw = switch_hand(scen)
    ok = lambda e: e["cell"] == truth_cell(scen, e["hand"]) and e["c"] > tau
    steady_end = sw if sw else hands
    pts = [ok(e) for e in r["log"] if start <= e["hand"] <= steady_end and e["hand"] > 0]
    acc = float(np.mean(pts)) if pts else None
    delay = None
    if sw:
        hit = next((e["hand"] for e in r["log"] if e["hand"] > sw and ok(e)), None)
        delay = (hit - sw) if hit is not None else hands - sw
    return acc, delay


def report(a):
    rows = []
    for lam in a.lams:
        acc, dly, by = [], [], {}
        for scen in a.scenarios:
            if scen == "FR_offGrid":
                continue
            for seed in range(a.seeds):
                f = sim_file(scen, seed, lam)
                if not os.path.exists(f):
                    continue
                ac, d = measure(json.load(open(f)), a.tau)
                if ac is not None:
                    acc.append(ac); by.setdefault(scen, []).append(ac)
                if d is not None:
                    dly.append(d)
        if acc:
            rows.append((lam, np.mean(acc), np.median(dly) if dly else float("nan"), by))
    print(f"confidence threshold tau = {a.tau}; seat 0 plays p = 0; {a.seeds} sessions per scenario\n")
    print(f"{'lam':>6}{'memory (hands)':>16}{'steady accuracy':>17}{'switch caught after':>21}")
    for lam, ac, d, _ in rows:
        mem = "all" if lam == 1.0 else f"~{1 / (1 - lam):.0f}"
        print(f"{lam:>6g}{mem:>16}{ac:>17.0%}{d:>17.0f} hands")
    base = rows[0][1] if rows and rows[0][0] == 1.0 else None
    if base is not None:
        ok = [r for r in rows if r[1] >= base - 0.05]
        best = min(ok, key=lambda r: r[2]) if ok else None
        if best:
            print(f"\nfastest memory whose steady accuracy is within 5 points of never forgetting: "
                  f"lam = {best[0]:g} (catches a switch after a median {best[2]:.0f} hands)")
    print("\nsteady accuracy by scenario (rows: lam):")
    scen = sorted({s for r in rows for s in r[3]})
    print(f"{'lam':>6}" + "".join(f"{s[:16]:>18}" for s in scen))
    for lam, _, _, by in rows:
        print(f"{lam:>6g}" + "".join(f"{np.mean(by[s]) if s in by else float('nan'):>18.0%}" for s in scen))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lams", type=float, nargs="+", default=LAMS)
    p.add_argument("--scenarios", nargs="+", default=list(ap.SCENARIOS))
    p.add_argument("--seeds", type=int, default=20); p.add_argument("--hands", type=int, default=300)
    p.add_argument("--every", type=int, default=5, help="hands between posterior readings")
    p.add_argument("--tau", type=float, default=0.5); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.report:
        jobs = [(s, sd, lam, a.hands, a.every) for s in a.scenarios for sd in range(a.seeds) for lam in a.lams]
        todo = [j for j in jobs if not os.path.exists(sim_file(j[0], j[1], j[2]))]
        print(f"{len(todo)} of {len(jobs)} sessions to simulate on {a.workers} workers", flush=True)
        t0 = time.time()
        with Pool(a.workers, initializer=ap.init_sim, initargs=(True,)) as pool:
            for i, (job, r) in enumerate(pool.imap_unordered(run, todo), 1):
                json.dump(r, open(sim_file(job[0], job[1], job[2]), "w"))
                if i % 100 == 0 or i == len(todo):
                    print(f"[{(time.time() - t0) / 60:6.1f} min] {i}/{len(todo)}", flush=True)
    report(a)


if __name__ == "__main__":
    main()
