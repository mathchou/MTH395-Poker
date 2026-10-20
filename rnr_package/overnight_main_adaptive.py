"""
Adaptive search over p for the main study's equal-bias settings (results/).

The main study mostly tested p = 0, 0.5 and 0.9, but on the independent-bias grid the safe
region typically ends at p = 0.6-0.8 -- between those points. This runs the same search as
stage 4 of overnight_grid.py (same rule and code) on the main study's settings:

  1. p = 0.25 wherever it is missing
  2. per setting: bisect on p between the largest safe p and the smallest unsafe p above it,
     until they are within 0.07, with at most 4 points beyond {0, 0.25, 0.5} (points already
     solved, such as p = 0.9, count toward the 4)

Results go to results/ with the standard names, so the existing tools pick them up. Because
reeval_pessimistic.py skips settings it has already scored, delete the affected
results/pess/pess_*_a<alpha>_*.json files afterwards and rerun it, so the headline (which uses
the pessimistic punisher) includes the new solves.

    python overnight_main_adaptive.py --list
    python overnight_main_adaptive.py                      # alpha 0.35, all seats (about a day)
    python overnight_main_adaptive.py --alpha 0.1 --seats 0
"""

import argparse
import os
import subprocess
import sys
import time

import overnight_grid as og

PY = sys.executable
RES = "results"
PAIRS = [a + b for a in "ORCF" for b in "ORCF" if a + b != "OO"]
ALPHA = "0.35"


def job(pair, seat, p=None):
    base = ["rnr_general.py", "--pair", pair, "--alpha", ALPHA, "--seat", str(seat)]
    if p is None:
        return f"M_linear_{pair}_a{ALPHA}_s{seat}", base + ["--linear"], f"{RES}/linear_{pair}_a{ALPHA}_s{seat}.json"
    return (f"M_{pair}_a{ALPHA}_s{seat}_p{p}",
            base + ["--p", str(p), "--restrict", "independent", "--free", "selfish", "--iters", "600"],
            f"{RES}/rnr_{pair}_a{ALPHA}_s{seat}_independent_selfish_p{p}.json")


def cell_points(pair, seat, files=None):
    """{p: raw worst case} for every un-suffixed solve of this setting in results/."""
    import json
    pre = f"rnr_{pair}_a{ALPHA}_s{seat}_independent_selfish_p"
    files = os.listdir(RES) if files is None else files
    pts = {}
    for f in files:
        if f.startswith(pre) and f.endswith(".json") and f[len(pre):-5].replace(".", "").isdigit():
            r = json.load(open(os.path.join(RES, f)))
            pts[r["p"]] = r["worst_selfish"]
    return pts


og.job, og.cell_points, og.MAX_EXTRA = job, cell_points, 4      # same search, main-study files


def main():
    global ALPHA
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default="0.35")
    ap.add_argument("--seats", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    ALPHA = str(float(a.alpha))
    files = os.listdir(RES)
    cells = [(s, pair) for s in a.seats for pair in PAIRS
             if os.path.exists(job(pair, s)[2]) and 0.0 in cell_points(pair, s, files)]
    static = [job(pair, s, 0.25) for s, pair in cells if 0.25 not in cell_points(pair, s, files)]
    static = [j for j in static if not os.path.exists(j[2])]
    eq = {s: og.seat_value(s) for s in a.seats}
    if a.list:
        fin, todo, wait, extra = og.stage4_progress(cells, eq, files)
        print(f"alpha {ALPHA}, seats {a.seats}: {len(cells)} settings; {len(static)} need p = 0.25 first")
        print(f"adaptive: {fin} settings finished, {todo} with a solve to run, {wait} waiting for p = 0.25 or 0.5; "
              f"{extra} points beyond 0, 0.25, 0.5 already solved")
        return
    os.makedirs(os.path.join("overnight", "logs"), exist_ok=True)
    running, t0, last = {}, time.time(), 0.0
    print(f"{len(static)} p = 0.25 solves, then adaptive search over {len(cells)} settings; "
          f"{a.workers} workers. Ctrl+C to stop; rerun to resume.", flush=True)

    def launch(name, argv):
        log = open(os.path.join("overnight", "logs", name + ".log"), "w")
        running[name] = (subprocess.Popen([PY, "-u"] + argv, stdout=log, stderr=subprocess.STDOUT), log)

    rescan, last_scan = True, 0.0
    try:
        while True:
            while static and len(running) < a.workers:
                name, argv, _ = static.pop(0); launch(name, argv)
            if len(running) < a.workers and (rescan or time.time() - last_scan > 120):
                rescan, last_scan = False, time.time()
                files = os.listdir(RES)
                for s, pair in cells:
                    if len(running) >= a.workers:
                        break
                    if any(n.startswith(f"M_{pair}_a{ALPHA}_s{s}_p") for n in running):
                        continue
                    p = og.next_p(pair, s, eq[s], cell_points(pair, s, files))
                    if p is not None:
                        name, argv, out = job(pair, s, p)
                        if not os.path.exists(out):
                            launch(name, argv)
            if not running:
                break
            time.sleep(5)
            for name in list(running):
                proc, log = running[name]
                if proc.poll() is not None:
                    log.close(); rescan = True; del running[name]
                    if proc.returncode != 0:
                        print(f"[{(time.time() - t0) / 60:6.1f} min] FAILED {name}", flush=True)
            if time.time() - last > 600:
                last = time.time()
                fin, todo, wait, extra = og.stage4_progress(cells, eq, os.listdir(RES))
                print(f"[{(time.time() - t0) / 60:6.1f} min] running {len(running)}; {fin} of {len(cells)} "
                      f"settings finished; {extra} adaptive points solved", flush=True)
        print(f"finished after {(time.time() - t0) / 60:.0f} min. Now delete results/pess/pess_*_a{ALPHA}_*.json "
              f"and rerun reeval_pessimistic.py so the pessimistic headline includes the new solves.")
    except KeyboardInterrupt:
        for proc, log in running.values():
            proc.terminate(); log.close()
        print("\nstopped; rerun to resume")


if __name__ == "__main__":
    main()
