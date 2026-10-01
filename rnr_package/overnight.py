"""
Overnight job runner. Runs everything the audit asked for, in priority order, on N parallel
workers. Resumable: finished jobs are skipped, so you can stop it in the morning (Ctrl+C) and
restart the next night with the same command. Every job logs to overnight/logs/.

    python overnight.py --list                       # show the plan and what is done
    python overnight.py                              # run with 8 workers
    python overnight.py --workers 6 --cfr1k ..\\alpha_study\\cfr_3p_1000.pkl

Groups, in the order they start:
  A  pessimistic re-evaluation of every result            (~1 h total, split across workers)
  C  both-opponents-adapt runs, logged to files           (5 jobs, ~15-30 min each)
  D  1k-era analyses rerun on the 5k baseline             (5 jobs)
  B  solution stability: 6 settings re-solved at 1200     (12 jobs, 40-100 min each)
     iterations and from a random start
  E  alpha 0.2 restricted responses, seat 0, all pairs    (75 jobs; fills the rest of the night)
"""

import argparse
import os
import subprocess
import sys
import time
from cfr_path import find_cfr

PY = sys.executable
LOGS = os.path.join("overnight", "logs")
DONE = os.path.join("overnight", "done")
PAIRS = [a + b for a in "ORCF" for b in "ORCF"]


def cfr5k():
    return find_cfr()


def jobs(nshards, cfr1k):
    J = []   # (name, argv, output_that_means_done_or_None)
    base = cfr5k()
    # A: pessimistic re-evaluation, sharded
    for k in range(nshards):
        argv = ["reeval_pessimistic.py", "--shard", str(k), "--nshards", str(nshards)]
        if cfr1k:
            argv += ["--cfr1k", cfr1k]
        J.append((f"A_pess_shard{k}of{nshards}", argv, None))
    # C: both-opponents-adapt (console output goes to the log)
    for name, extra in [("cfr_uniform", []), ("cfr_seed1", ["--seed", "1"]), ("cfr_seed3", ["--seed", "3"]),
                        ("rnr05_uniform", []), ("rnr09_uniform", [])]:
        strat = {"rnr05": "RNR p=0.5", "rnr09": "RNR p=0.9"}.get(name.split("_")[0], "CFR")
        J.append((f"C_bothadapt_{name}",
                  ["gaps_check.py", "bothadapt", "--iters", "500", "--strategies", strat] + extra, None))
    # D: 1k-era analyses on the 5k baseline
    J.append(("D_q1_alpha_sweep_5k", ["q1_alpha_sweep.py", "--cfr", base, "--cache", "q1_results_5k.json",
                                      "--save-br", "0.1", "0.35"], None))
    for a in ("0.1", "0.2", "0.35"):
        J.append((f"D_q2_identify_5k_a{a}", ["q2_identify.py", "--cfr", base, "--alpha", a, "--pairs", *PAIRS,
                                              "--sessions", "50", "--hands", "250", "--out", f"q2_5k_a{a}.json"],
                  f"q2_5k_a{a}.json"))
    J.append(("D_q6_equal_deviation_5k", ["q6_equal_tv.py", "--cfr", base, "--m", "0.05", "0.1", "0.15", "0.2", "0.25"], None))
    # B: solution stability
    for pair, a, s, p in [("FF", 0.35, 0, 0.5), ("RR", 0.35, 0, 0.5), ("CC", 0.35, 0, 0.5),
                          ("RO", 0.35, 0, 0.25), ("OF", 0.35, 0, 0.5), ("RR", 0.35, 2, 0.5)]:
        common = ["rnr_general.py", "--pair", pair, "--alpha", str(a), "--seat", str(s), "--p", str(p),
                  "--restrict", "independent", "--free", "selfish"]
        tag = f"{pair}_a{a}_s{s}_independent_selfish_p{p}"
        J.append((f"B_{tag}_it1200", common + ["--iters", "1200", "--suffix", "it1200"],
                  f"results/rnr_{tag}_it1200.json"))
        J.append((f"B_{tag}_seed1", common + ["--iters", "600", "--seed", "1", "--suffix", "seed1"],
                  f"results/rnr_{tag}_seed1.json"))
    # E: alpha 0.2 grid, seat 0 (p = 0 reuses the cached solve and takes about a minute)
    for pair in PAIRS:
        J.append((f"E_linear_{pair}_a0.2", ["rnr_general.py", "--pair", pair, "--alpha", "0.2", "--seat", "0", "--linear"],
                  f"results/linear_{pair}_a0.2_s0.json"))
        if pair == "OO":
            continue
        for p in (0.0, 0.25, 0.5, 0.9):
            J.append((f"E_{pair}_a0.2_p{p}", ["rnr_general.py", "--pair", pair, "--alpha", "0.2", "--seat", "0",
                                              "--p", str(p), "--restrict", "independent", "--free", "selfish",
                                              "--iters", "600"],
                      f"results/rnr_{pair}_a0.2_s0_independent_selfish_p{p}.json"))
    return J


def is_done(name, out):
    return os.path.exists(os.path.join(DONE, name)) or (out is not None and os.path.exists(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cfr1k", default=None, help="path to the 1,000-iteration CFR table (optional)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if not os.path.exists("leduc3p_tree.npz"):
        subprocess.run([PY, "tree_engine.py", "build"], check=True)
    if a.cfr1k and not os.path.exists(a.cfr1k):
        raise SystemExit(f"--cfr1k file not found: {a.cfr1k}")
    os.makedirs(LOGS, exist_ok=True); os.makedirs(DONE, exist_ok=True)
    todo = jobs(a.workers, a.cfr1k)
    if a.list:
        for name, argv, out in todo:
            print(f"{'DONE' if is_done(name, out) else 'todo'}  {name}")
        print(f"\n{sum(1 for n, _, o in todo if not is_done(n, o))} of {len(todo)} jobs to run")
        return
    queue = [j for j in todo if not is_done(j[0], j[2])]
    running = {}
    t0 = time.time()
    print(f"{len(queue)} jobs, {a.workers} workers. Ctrl+C to stop; rerun to resume.", flush=True)
    try:
        while queue or running:
            while queue and len(running) < a.workers:
                name, argv, out = queue.pop(0)
                log = open(os.path.join(LOGS, name + ".log"), "w")
                running[name] = (subprocess.Popen([PY, "-u"] + argv, stdout=log, stderr=subprocess.STDOUT), log, out)
                print(f"[{(time.time()-t0)/60:6.1f} min] start {name}", flush=True)
            time.sleep(5)
            for name in list(running):
                proc, log, out = running[name]
                if proc.poll() is not None:
                    log.close()
                    ok = proc.returncode == 0
                    if ok:
                        open(os.path.join(DONE, name), "w").write("done")
                    print(f"[{(time.time()-t0)/60:6.1f} min] {'done  ' if ok else 'FAILED'} {name}"
                          + ("" if ok else f"  (see {LOGS}/{name}.log)"), flush=True)
                    del running[name]
    except KeyboardInterrupt:
        print("\nstopping: terminating running jobs (they will restart next time)")
        for proc, log, _ in running.values():
            proc.terminate(); log.close()


if __name__ == "__main__":
    main()
