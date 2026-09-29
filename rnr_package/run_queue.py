"""
Resumable job queue for the RNR experiments. Skips any job whose result file exists,
so you can stop and restart freely. Cross-platform (works in PowerShell).

    python run_queue.py --priority 1                  # run priority-1 jobs in order
    python run_queue.py --priority 1 2 3 4            # everything
    python run_queue.py --priority 3 --shard 0 --nshards 2   # split across 2 terminals
    python run_queue.py --list                        # show jobs and status

Each RNR job takes roughly 10-15 minutes at 600 iterations (independent restriction);
linear baselines take 2-3 minutes. Each process needs about 0.5 GB of RAM.
"""
import argparse
import os
import subprocess
import sys

P_GRID = [0.0, 0.25, 0.5, 0.75, 0.9, 0.97]   # fine grid, headline pair
P_BROAD = [0.0, 0.5, 0.9]                     # robust end, knee, near best response
ITERS = 600
PAIRS = [a + b for a in "ORCF" for b in "ORCF"]  # all 16 ORDERED pairs; OO is a control


def jobs():
    J = []
    # 1: RERUN-21/22 core — symmetric, self-interested, the headline experiment
    J += [(1, "FF", 0.35, 0, "independent", "selfish", p) for p in P_GRID]
    J += [(1, "FF", 0.35, 0, "linear", None, None)]
    # 2: RERUN-22 comparison — one-sided with a self-interested free opponent
    J += [(2, "FF", 0.35, 0, "one-sided", "selfish", p) for p in [0.25, 0.5, 0.75]]
    # 3: RERUN-10 — every ordered pair at alpha 0.35, seat 0
    #    (FF's p = 0, 0.5, 0.9 already come from priority 1 and are skipped)
    for pair in PAIRS:
        J += [(3, pair, 0.35, 0, "independent", "selfish", p) for p in P_BROAD]
        J += [(3, pair, 0.35, 0, "linear", None, None)]
    # 4: RERUN-16 — WE sit in seat 1 or 2. Headline pair on the fine grid, plus a
    #    representative set on the broad grid: both single-type pairs that behave most
    #    differently by seat (RR, CC), both orders of a mixed pair (FR, RF), and the
    #    OO control. The rest of the seat-1/2 grid is priority 6.
    for seat in [1, 2]:
        J += [(4, "FF", 0.35, seat, "independent", "selfish", p) for p in P_GRID[:5]]
        J += [(4, "FF", 0.35, seat, "linear", None, None)]
        for pair in ["RR", "CC", "FR", "RF", "OO"]:
            J += [(4, pair, 0.35, seat, "independent", "selfish", p) for p in P_BROAD]
            J += [(4, pair, 0.35, seat, "linear", None, None)]
    # 5: every ordered pair at alpha 0.1, seat 0
    for pair in PAIRS:
        J += [(5, pair, 0.1, 0, "independent", "selfish", p) for p in P_BROAD]
        J += [(5, pair, 0.1, 0, "linear", None, None)]
    # 6: every ordered pair at alpha 0.35, seats 1 and 2 (full seat coverage)
    for seat in [1, 2]:
        for pair in PAIRS:
            J += [(6, pair, 0.35, seat, "independent", "selfish", p) for p in P_BROAD]
            J += [(6, pair, 0.35, seat, "linear", None, None)]
    # 7: the 12 settings where no tested p beat CFR on both axes. All were tested only at
    #    p in {0, 0.5, 0.9}; safe wins usually sit at low p (RO seat 0 won at p = 0.25).
    NO_WIN = ([(pair, 0.1, 0) for pair in ["CR", "FC", "FO", "OF", "OR", "RC", "RR"]]
              + [("FO", 0.35, 0), ("FO", 0.35, 2), ("OF", 0.35, 2), ("OR", 0.35, 2), ("RO", 0.35, 2)])
    for pair, a, seat in NO_WIN:
        J += [(7, pair, a, seat, "independent", "selfish", p) for p in (0.1, 0.25)]
    # de-duplicate (a job listed under two priorities runs once, under the first)
    seen, out = set(), []
    for j in J:
        k = j[1:]
        if k not in seen:
            seen.add(k); out.append(j)
    return out


def outfile(j):
    pr, pair, a, seat, restrict, free, p = j
    if restrict == "linear":
        return f"results/linear_{pair}_a{a}_s{seat}.json"
    return f"results/rnr_{pair}_a{a}_s{seat}_{restrict}_{free}_p{p}.json"


def cmd(j):
    pr, pair, a, seat, restrict, free, p = j
    c = [sys.executable, "-u", "rnr_general.py", "--pair", pair, "--alpha", str(a),
         "--seat", str(seat)]
    if restrict == "linear":
        return c + ["--linear"]
    return c + ["--p", str(p), "--restrict", restrict, "--free", free, "--iters", str(ITERS)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--priority", nargs="+", type=int, default=[1])
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if not a.list and not os.path.exists("leduc3p_tree.npz"):
        print("building tree (one-time, ~2 min)...", flush=True)
        subprocess.run([sys.executable, "tree_engine.py", "build"], check=True)
    todo = [j for j in jobs() if j[0] in a.priority]
    todo = [j for i, j in enumerate(todo) if i % a.nshards == a.shard]
    for j in todo:
        done = os.path.exists(outfile(j))
        if a.list:
            print(f"{'DONE' if done else 'todo'}  {outfile(j)}")
            continue
        if done:
            continue
        print(f"\n=== {outfile(j)}", flush=True)
        os.makedirs("results", exist_ok=True)
        with open(outfile(j).replace(".json", ".log"), "w") as log:
            r = subprocess.run(cmd(j), stdout=log, stderr=subprocess.STDOUT)
        print("ok" if r.returncode == 0 else f"FAILED (see {outfile(j).replace('.json', '.log')})",
              flush=True)