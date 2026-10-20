"""
The chase, swept: three adaptive players under different memories and update intervals.

Each session runs session_sim.py with every seat adaptive (by default starting as an over-folder,
an over-raiser and an over-caller at alpha 0.3) and a given way of letting old evidence fade:

    none          stationary identifier (the original chase)
    forget M      exponential forgetting, memory ~ M hands
    switch M      fixed-share, opponents expected to change about every M hands

crossed with how often the players update their strategies (--every), over --seeds sessions.
Sessions run in parallel; finished ones are skipped, so it is resumable. Results: chase/*.json.

Report, per (method, update interval), medians over sessions of second-half measures:
    settled     share of updates in which a player's strategy barely moved (change < 0.01)
    volatility  mean absolute change in a seat's exact EV from one update to the next
    confidence  average confidence c of the three players
    EV seat k   each seat's average exact EV relative to equilibrium

    python chase_sweep.py --list
    python chase_sweep.py                          # 540 sessions, ~3 h on 8 workers
    python chase_sweep.py --report
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

OUT = "chase"
PY = sys.executable


def configs(a):
    methods = [("none", [])] + [(f"forget{m}", ["--forget", str(1 - 1 / m)]) for m in a.memories] + \
              [(f"switch{m}", ["--switch", str(1 / m)]) for m in a.memories]
    J = []
    for name, extra in methods:
        for every in a.every:
            for seed in range(a.seeds):
                out = os.path.join(OUT, f"{name}_e{every}_s{seed}.json")
                argv = ["session_sim.py", "--seat0", a.seats[0], "--seat1", a.seats[1], "--seat2", a.seats[2],
                        "--hands", str(a.hands), "--every", str(every), "--beta", str(a.beta),
                        "--seed", str(seed), "--out", out] + extra
                J.append((f"{name}_e{every}_s{seed}", argv, out))
    return J


def summarise(path):
    tr = json.load(open(path))["sessions"][0]["trace"]
    half = [r for r in tr if r["hand"] > tr[-1]["hand"] / 2]
    ev = np.array([r["ev"] for r in half])
    ch = [ag["change"] for r in half[1:] for ag in r["agents"] if "change" in ag and "conf" in ag]
    cf = [ag["conf"] for r in half for ag in r["agents"] if "conf" in ag]
    return dict(settled=float(np.mean(np.array(ch) < 0.01)) if ch else np.nan,
                volatility=float(np.mean(np.abs(np.diff(ev, axis=0)))) if len(ev) > 1 else np.nan,
                confidence=float(np.mean(cf)) if cf else np.nan, ev=ev.mean(0).tolist())


def report(a):
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)       # blank cells when a session is very short
    rows = {}
    for f in glob.glob(os.path.join(OUT, "*.json")):
        name = os.path.basename(f)[:-5]
        method, every, _ = name.rsplit("_", 2)
        rows.setdefault((method, int(every[1:])), []).append(summarise(f))
    if not rows:
        raise SystemExit(f"no sessions in {OUT}/ yet")
    order = lambda k: ({"none": 0, "forget": 1, "switch": 2}[k[0].rstrip("0123456789")],
                       int("".join(ch for ch in k[0] if ch.isdigit()) or 0), k[1])
    print(f"{'method':<11}{'every':>6}{'n':>4}{'settled':>9}{'volatility':>12}{'confidence':>12}"
          f"{'EV seat 0':>11}{'EV seat 1':>11}{'EV seat 2':>11}")
    for k in sorted(rows, key=order):
        R = rows[k]; med = lambda key: np.nanmedian([r[key] for r in R])
        evs = np.median(np.array([r["ev"] for r in R]), axis=0)
        print(f"{k[0]:<11}{k[1]:>6}{len(R):>4}{med('settled'):>9.2f}{med('volatility'):>12.3f}"
              f"{med('confidence'):>12.2f}" + "".join(f"{e:>+11.3f}" for e in evs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seats", nargs=3, default=["adaptive:F:0.3", "adaptive:R:0.3", "adaptive:C:0.3"])
    ap.add_argument("--memories", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--every", type=int, nargs="+", default=[10, 25, 50])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--hands", type=int, default=400)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--list", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        return report(a)
    J = configs(a)
    todo = [j for j in J if not os.path.exists(j[2])]
    if a.list:
        print(f"{len(J)} sessions, {len(todo)} to run")
        return
    os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
    running, t0, done = {}, time.time(), 0
    print(f"{len(todo)} sessions on {a.workers} workers. Ctrl+C to stop; rerun to resume.", flush=True)
    try:
        while todo or running:
            while todo and len(running) < a.workers:
                name, argv, out = todo.pop(0)
                log = open(os.path.join(OUT, "logs", name + ".log"), "w")
                running[name] = (subprocess.Popen([PY, "-u"] + argv, stdout=log, stderr=subprocess.STDOUT), log)
            time.sleep(5)
            for name in list(running):
                proc, log = running[name]
                if proc.poll() is not None:
                    log.close(); del running[name]; done += 1
                    if proc.returncode != 0:
                        print(f"FAILED {name} (see {OUT}/logs/{name}.log)", flush=True)
                    elif done % 20 == 0:
                        print(f"[{(time.time() - t0) / 60:6.1f} min] {done} finished, {len(todo)} queued", flush=True)
        print(f"finished after {(time.time() - t0) / 60:.0f} min; now: python chase_sweep.py --report")
    except KeyboardInterrupt:
        for proc, log in running.values():
            proc.terminate(); log.close()
        print("\nstopped; rerun to resume")


if __name__ == "__main__":
    main()
