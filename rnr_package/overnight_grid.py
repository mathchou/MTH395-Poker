"""
Restricted-response runs over independent biases (alpha1, alpha2), seat 0.

At alpha = 0 every type IS the equilibrium player, so a table with a zero is shared by
every pair's grid; each distinct table is computed once (written with type O).

  stage 1  linear baselines, FINE alpha grid (0, 0.05, ..., 0.6)        1,369 tables, ~9 h
  stage 2  p = 0 (cached solve), FINE alpha grid                        1,369 tables, ~3 h
  stage 3  p = 0.25 and 0.5, COARSE alpha grid (0, 0.1, ..., 0.6)       720 solves,   ~90 h
  stage 4  adaptive p per coarse cell: find where the safe region ends   ~2-3 solves per cell
           (hours for 8 workers at 600 iterations)

Stages 1-2 need no full solves, so one night maps the whole fine-alpha landscape (best-
response gain, how punishable equilibrium play is). If it is smooth between coarse grid
points, coarse alpha is enough for the expensive solves.

Stage 4, per cell. "Safe" means the worst case is no more than 0.05 below the best
equilibrium-level strategy (CFR or p = 0), as in the paper; p = 0 counts as safe.
  lo = largest p known to be safe;  hi = smallest p above lo known to be unsafe
  if no unsafe p above lo:  try 0.9 (if lo < 0.9), else stop
  otherwise bisect until hi - lo <= 0.07, at most 3 extra solves per cell
Safety is NOT always monotone in p (RR seat 0 is unsafe at 0.25 but safe at 0.5), so the
search tracks the LARGEST safe p seen, and only looks above it. Worst cases here use the
exact punisher stored in each result.

Resumable: stop with Ctrl+C, rerun the same command. Results go to results_het/.

    python overnight_grid.py --list                   # progress by stage
    python overnight_grid.py                          # all stages, in order
    python overnight_grid.py --stages 1 2             # just the cheap fine-grid maps
    python overnight_grid.py --pairs FR RF            # only the grids for some pairs
    python analyze_het.py                             # tables and heatmaps, any time
"""

import argparse
import json
import os
import subprocess
import sys
import time

PY = sys.executable
LOGS = os.path.join("overnight", "logs")
DONE = os.path.join("overnight", "done")
HET = "results_het"
FINE = [round(0.05 * k, 2) for k in range(1, 13)]          # 0.05 ... 0.6
COARSE = [round(0.1 * k, 1) for k in range(1, 7)]          # 0.1 ... 0.6
TYPES = "RCF"
TOL_WORST, TOL_P, MAX_EXTRA, TOP_P = 0.05, 0.07, 3, 0.9
BASE_PS = {0.0, 0.25, 0.5}
ITERS = "600"


# ---------------------------------------------------------------- the grid

def settings(alphas):
    """Distinct settings for one opponent: equilibrium, then each type at each alpha."""
    return [("O", 0.0)] + [(t, a) for t in TYPES for a in alphas]


def in_pairs(cfg, pairs):
    (t1, _), (t2, _) = cfg
    return pairs is None or any(t1 in ("O", X) and t2 in ("O", Y) for X, Y in pairs)


def is_oo(cfg):
    return cfg[0][0] == "O" and cfg[1][0] == "O"


def label(cfg):
    (t1, a1), (t2, a2) = cfg
    return t1 + t2, (f"{a1}" if a1 == a2 else f"{a1}-{a2}")


def job(cfg, p=None):
    """(name, command, output file) for the linear baseline (p None) or a solve at p."""
    (t1, a1), (t2, a2) = cfg
    pair, tag = label(cfg)
    base = ["rnr_general.py", "--pair", pair, "--alpha", str(a1), "--alpha2", str(a2), "--seat", "0",
            "--out", HET, "--cache-dir", "results"]
    if p is None:
        return f"G_linear_{pair}_a{tag}", base + ["--linear"], f"{HET}/linear_{pair}_a{tag}_s0.json"
    return (f"G_{pair}_a{tag}_p{p}",
            base + ["--p", str(p), "--restrict", "independent", "--free", "selfish", "--iters", ITERS],
            f"{HET}/rnr_{pair}_a{tag}_s0_independent_selfish_p{p}.json")


def static_jobs(stages, pairs):
    fine = [(x, y) for x in settings(FINE) for y in settings(FINE) if in_pairs((x, y), pairs)]
    coarse = [(x, y) for x in settings(COARSE) for y in settings(COARSE) if in_pairs((x, y), pairs)]
    J = []
    if 1 in stages: J += [(1,) + job(c) for c in fine]
    if 2 in stages: J += [(2,) + job(c, 0.0) for c in fine]
    if 3 in stages: J += [(3,) + job(c, p) for c in coarse if not is_oo(c) for p in (0.25, 0.5)]
    return J, [c for c in coarse if not is_oo(c)]


def done(name, out):
    return os.path.exists(os.path.join(DONE, name)) or os.path.exists(out)


# ---------------------------------------------------------------- stage 4: adaptive p

def seat0_value():
    f = "results/linear_OO_a0.35_s0.json"
    if os.path.exists(f):
        return json.load(open(f))["rows"][0]["vs_biased"]
    import pickle
    from tree_engine import Tree
    from cfr_path import find_cfr
    T = Tree(); base = pickle.load(open(find_cfr(), "rb"))["table"]
    return T.ev([T.from_table(base, q) for q in range(3)])[0]


def cell_points(cfg, files=None):
    """{p: worst case (raw)} for every finished solve of this cell. Pass `files` (a listing of
    results_het/) to avoid listing the folder once per cell."""
    pair, tag = label(cfg)
    pre = f"rnr_{pair}_a{tag}_s0_independent_selfish_p"
    pts = {}
    if files is None:
        files = os.listdir(HET) if os.path.isdir(HET) else []
    if True:
        for f in files:
            if f.startswith(pre) and f.endswith(".json") and f[len(pre):-5].replace(".", "").isdigit():
                r = json.load(open(os.path.join(HET, f)))
                pts[r["p"]] = r["worst_selfish"]
    return pts


def next_p(cfg, eq, pts=None):
    """The next p to solve for this cell, or None (waiting for stages 1-3, or finished)."""
    lin = job(cfg)[2]
    pts = cell_points(cfg) if pts is None else pts
    if not os.path.exists(lin) or not BASE_PS <= set(pts):
        return None
    if len(set(pts) - BASE_PS) >= MAX_EXTRA:
        return None
    rows = json.load(open(lin))["rows"]
    wb = max(rows[0]["worst_selfish"], pts[0.0]) - eq          # best equilibrium-level worst case
    safe = {p: (w - eq >= wb - TOL_WORST) for p, w in pts.items()}
    safe[0.0] = True
    lo = max(p for p, s in safe.items() if s)
    above = [p for p, s in safe.items() if not s and p > lo]
    if not above:
        return TOP_P if lo < TOP_P and TOP_P not in pts else None
    hi = min(above)
    if hi - lo <= TOL_P:
        return None
    p = round((lo + hi) / 2, 3)
    return p if p not in pts else None


# ---------------------------------------------------------------- runner

def main():
    global ITERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--pairs", nargs="+", default=None, help="e.g. FR RF (default: all)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    ITERS = str(a.iters)
    stages = set(a.stages)
    pairs = [tuple(p) for p in a.pairs] if a.pairs else None
    if not os.path.exists("leduc3p_tree.npz"):
        subprocess.run([PY, "tree_engine.py", "build"], check=True)
    if not os.path.exists(os.path.join("results", "p0cache_s0_selfish.npz")):
        print("note: no p = 0 cache in results/; the first p = 0 job will solve it (~20-50 min)")
    os.makedirs(LOGS, exist_ok=True); os.makedirs(DONE, exist_ok=True); os.makedirs(HET, exist_ok=True)
    static, cells = static_jobs(stages, pairs)
    eq = seat0_value() if 4 in stages else None

    if a.list:
        files = os.listdir(HET)
        for st in sorted({j[0] for j in static}):
            S = [j for j in static if j[0] == st]
            print(f"stage {st}: {sum(done(n, o) for _, n, _, o in S)} of {len(S)} done")
        if 4 in stages:
            fin = wait = todo = 0
            for c in cells:
                pts = cell_points(c, files)
                ready = os.path.exists(job(c)[2]) and BASE_PS <= set(pts)
                n = next_p(c, eq, pts) if ready else None
                fin += ready and n is None; wait += not ready; todo += ready and n is not None
            print(f"stage 4: {fin} of {len(cells)} cells finished, {todo} with a solve to run, "
                  f"{wait} waiting for stages 1-3")
        return

    queue = [j for j in static if not done(j[1], j[3])]
    running, t0, last = {}, time.time(), 0.0
    rescan, last_scan = True, 0.0           # stage 4 only rescans when something finished
    print(f"{len(queue)} static jobs (stages {sorted(stages & {1, 2, 3})})"
          + (f" + adaptive stage 4 over {len(cells)} cells" if 4 in stages else "")
          + f", {a.workers} workers. Ctrl+C to stop; rerun to resume.", flush=True)

    def launch(name, argv):
        log = open(os.path.join(LOGS, name + ".log"), "w")
        running[name] = (subprocess.Popen([PY, "-u"] + argv, stdout=log, stderr=subprocess.STDOUT), log)

    try:
        while True:
            while queue and len(running) < a.workers:              # static jobs first
                _, name, argv, _ = queue.pop(0); launch(name, argv)
            if 4 in stages and len(running) < a.workers and (rescan or time.time() - last_scan > 120):
                rescan, last_scan = False, time.time()
                files = os.listdir(HET)
                for c in cells:
                    if len(running) >= a.workers:
                        break
                    pair, tag = label(c)
                    if any(n.startswith(f"G_{pair}_a{tag}_p") for n in running):
                        continue                                    # one solve per cell at a time
                    p = next_p(c, eq, cell_points(c, files))
                    if p is not None:
                        name, argv, out = job(c, p)
                        if not done(name, out):
                            launch(name, argv)
            if not running:
                break
            time.sleep(5)
            for name in list(running):
                proc, log = running[name]
                if proc.poll() is not None:
                    log.close(); rescan = True
                    if proc.returncode == 0:
                        open(os.path.join(DONE, name), "w").write("done")
                    else:
                        print(f"[{(time.time()-t0)/60:6.1f} min] FAILED {name} (see {LOGS}/{name}.log)", flush=True)
                    del running[name]
            if time.time() - last > 600:                            # progress every 10 minutes
                last = time.time()
                left = sum(not done(n, o) for _, n, _, o in static)
                print(f"[{(time.time()-t0)/60:6.1f} min] static jobs left: {left}; running: {len(running)}",
                      flush=True)
        print(f"finished after {(time.time()-t0)/60:.0f} min")
    except KeyboardInterrupt:
        print("\nstopping: running jobs will restart next time")
        for proc, log in running.values():
            proc.terminate(); log.close()


if __name__ == "__main__":
    main()
