"""
Is the true model's own restricted-response curve really beaten by other bank strategies, or is
that a convergence artifact?

For each case, re-solve (seat 0)
    the TRUE model's own points  (the cell matching the true opponents, at the given p values)
    the WINNER                   (the best safe bank strategy against those opponents, found by
                                  adapt_policies.py's hindsight oracle)
three more ways each:
    it1200   twice the iterations
    seed1    a different random start
    seed2    another random start
then score every version (and the originals) against the TRUE opponents: expected payoff and the
pessimistic worst case, both relative to the best equilibrium-level strategy (CFR or p = 0).

If more iterations or another start lifted the true model's points to the winner's level, the
finding would be an artifact. Re-solves go to results_het_check/.

    python own_frontier_check.py --list
    python own_frontier_check.py              # solves, then scores (~3-4 h on 8 workers)
    python own_frontier_check.py --score      # score whatever has finished
"""

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np

CHECK = "results_het_check"
PY = sys.executable
TOL = 0.05
CASES = {     # true opponents: (the true model's own p values, the winner as cell:p)
    "F0.4-R0.2": ([0.375, 0.5], "F0.2-R0.4:0.7"),
    "R0.4-O0": ([0.312, 0.375], "R0.4-R0.2:0.7"),
}
VARIANTS = [("it1200", ["--iters", "1200"]), ("seed1", ["--iters", "600", "--seed", "1"]),
            ("seed2", ["--iters", "600", "--seed", "2"])]


def parse_cell(s):
    m = re.match(r"([ORCF])([0-9.]+)-([ORCF])([0-9.]+)$", s)
    return m.group(1), float(m.group(2)), m.group(3), float(m.group(4))


def files(cell, p, suffix=None):
    """(result json, strategy file, rnr_general command) for a cell at p; suffix = a re-solve variant.
    Names follow rnr_general.py: alphas as Python prints floats, e.g. F0.4-R0.2 -> FR_a0.4-0.2."""
    t1, a1, t2, a2 = parse_cell(cell)
    a1s, a2s = str(a1), str(a2)                          # 0.4 -> '0.4', 0 -> '0.0'
    tag = a1s if a1s == a2s else f"{a1s}-{a2s}"
    folder = CHECK if suffix else "results_het"
    base = f"{folder}/rnr_{t1}{t2}_a{tag}_s0_independent_selfish_p{p}" + (f"_{suffix}" if suffix else "")
    argv = ["rnr_general.py", "--pair", t1 + t2, "--alpha", a1s, "--alpha2", a2s, "--seat", "0", "--p", str(p),
            "--restrict", "independent", "--free", "selfish", "--out", CHECK]
    return base + ".json", base + "_strategy.npy", argv


def strategies():
    """[(case truth, label, cell, p)] for every strategy to re-solve."""
    out = []
    for truth, (own_ps, winner) in CASES.items():
        out += [(truth, f"own p={p:g}", truth, p) for p in own_ps]
        wc, wp = winner.split(":")
        out.append((truth, f"winner {wc} p={float(wp):g}", wc, float(wp)))
    return out


def jobs():
    J = []
    for truth, label, cell, p in strategies():
        for suf, extra in VARIANTS:
            js, npy, argv = files(cell, p, suf)
            J.append((f"{cell}_p{p}_{suf}", argv + extra + ["--suffix", suf], npy))
    return J


def score():
    import adapt_policies as ap
    from reeval_pessimistic import respond_pess
    ap.init_sim()
    T, S = ap.G["T"], ap.G["S"]
    eq = T.ev(S.base)[0]

    def evaluate(sig, truth):
        """(EV, pessimistic worst case) of seat-0 strategy sig against the true opponents, vs equilibrium."""
        t1, a1, t2, a2 = parse_cell(truth)
        opp = {1: S.tilt(1, t1, a1), 2: S.tilt(2, t2, a2)}
        worst = []
        for o, other in ((1, 2), (2, 1)):            # one punisher at a time; the other keeps its style
            pr = [None] * 3; pr[0], pr[other] = sig, opp[other]
            worst.append(respond_pess(T, pr, o, T.util[:, o], T.util[:, 0], 0.01))
        return T.ev([sig, opp[1], opp[2]])[0] - eq, min(worst) - eq

    for truth in CASES:
        p0 = next(iter(c for c in ap.G["bank"].values() if 0.0 in c["paths"]))["paths"][0.0]
        base = [evaluate(S.base[0], truth), evaluate(np.load(p0), truth)]
        gb, wb = max(b[0] for b in base), max(b[1] for b in base)
        print(f"\ntrue opponents {truth}: best equilibrium-level strategy earns {gb:+.3f}, worst {wb:+.3f}")
        print(f"  {'strategy':<28}{'version':<10}{'profit vs base':>15}{'safety margin':>15}  safe?")
        for t, label, cell, p in strategies():
            if t != truth:
                continue
            for suf in ["original"] + [v for v, _ in VARIANTS]:
                npy = files(cell, p, None if suf == "original" else suf)[1]
                if not os.path.exists(npy):
                    print(f"  {label:<28}{suf:<10}{'(not solved yet)':>30}")
                    continue
                g, w = evaluate(np.load(npy), truth)
                print(f"  {label:<28}{suf:<10}{g - gb:>+15.3f}{w - wb:>+15.3f}  {'yes' if w - wb >= -TOL else 'NO'}")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--list", action="store_true"); a.add_argument("--score", action="store_true")
    a = a.parse_args()
    J = jobs()
    todo = [j for j in J if not os.path.exists(j[2])]
    if a.list:
        for truth, label, cell, p in strategies():
            print(f"{truth}: {label}  (original: {'found' if os.path.exists(files(cell, p)[1]) else 'MISSING'})")
        print(f"{len(J)} re-solves, {len(todo)} still to run")
        return
    if not a.score and todo:
        os.makedirs(os.path.join(CHECK, "logs"), exist_ok=True)
        running, t0 = {}, time.time()
        print(f"{len(todo)} re-solves on {a.workers} workers", flush=True)
        try:
            while todo or running:
                while todo and len(running) < a.workers:
                    name, argv, out = todo.pop(0)
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
            print("\nstopped; rerun to resume"); return
    score()


if __name__ == "__main__":
    main()
