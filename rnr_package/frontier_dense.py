"""
Dense frontier curves for the paper's illustrative cells (seat 0): profit vs worst case as p rises.

The grid's adaptive search places ~5 points per cell, enough to find where safety ends but sparse
for a figure meant to show the abrupt switch. This solves p = 0.1, 0.2, ..., 0.9 for:
    FF 0.4/0.4    two deviators
    F 0.4 alone   a lone over-folder beside an equilibrium player
    C 0.4 alone   a lone over-caller beside an equilibrium player
Output goes to results_het/ under the standard names, so every other script uses the extra points.
Existing solves are skipped (resumable). ~27 solves, about 3.5 h on 8 workers.

    python frontier_dense.py --list
    python frontier_dense.py
"""

import argparse
import os
import subprocess
import sys
import time

PY = sys.executable
CELLS = [("FF", "0.4", "0.4"), ("FO", "0.4", "0.0"), ("CO", "0.4", "0.0")]
PS = [round(0.1 * k, 1) for k in range(1, 10)]


def jobs(iters):
    J = []
    for pair, a1, a2 in CELLS:
        tag = a1 if a1 == a2 else f"{a1}-{a2}"
        for p in PS:
            out = f"results_het/rnr_{pair}_a{tag}_s0_independent_selfish_p{p}.json"
            argv = ["rnr_general.py", "--pair", pair, "--alpha", a1, "--alpha2", a2, "--seat", "0", "--p", str(p),
                    "--restrict", "independent", "--free", "selfish", "--iters", str(iters),
                    "--out", "results_het", "--cache-dir", "results"]
            J.append((f"D_{pair}_a{tag}_s0_p{p}", argv, out))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    todo = [j for j in jobs(a.iters) if not os.path.exists(j[2])]
    if a.list:
        print(f"{len(todo)} of {len(jobs(a.iters))} solves still to run")
        return
    os.makedirs(os.path.join("overnight", "logs"), exist_ok=True)
    running, t0 = {}, time.time()
    print(f"{len(todo)} solves on {a.workers} workers", flush=True)
    try:
        while todo or running:
            while todo and len(running) < a.workers:
                name, argv, out = todo.pop(0)
                log = open(os.path.join("overnight", "logs", name + ".log"), "w")
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
