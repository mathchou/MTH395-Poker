"""
Two robustness checks for RNR strategies (run from inside rnr_package/).

    python gaps_check.py misspec
    python gaps_check.py bothadapt --iters 1000 --strategies CFR
    python gaps_check.py bothadapt --iters 1000 --strategies CFR --seed 1
    python gaps_check.py bothadapt --iters 1000 --strategies CFR "RNR p=0.5" --report-every 100

misspec    Strategies tuned for FF at alpha 0.35 (seat 0), evaluated against
           opponents they were not tuned for.

bothadapt  Our strategy is held fixed in seat 0. Seats 1 and 2 BOTH learn at once
           (CFR+, each maximising its own payoff, no collusion). Reports our EV
           relative to equilibrium as they learn, and each opponent's best-response
           gap (how far they are from a stable point). If the gaps shrink toward 0
           while our EV stays well below 0, the opponents have found a second
           equilibrium that is worse for us. If our EV drifts back toward 0, the
           earlier result was transient.

           --seed N starts both opponents from a random strategy instead of uniform.
           Run several seeds to see whether they always land in the same place.

Needs: tree_engine.py, q1_alpha_sweep.py, leduc3p_tree.npz (python tree_engine.py
build), the CFR table, and for RNR strategies the files in results/.
"""

import argparse
import os
import pickle
import time

import numpy as np

from tree_engine import Tree
from q1_alpha_sweep import perturb

RESULTS = "results"
CFR_CANDIDATES = ["cfr_3p_5000.pkl", "cfr_3p_1000.pkl"]   # 5k contents may be saved under either name


def load_base():
    for f in CFR_CANDIDATES:
        if os.path.exists(f):
            blob = pickle.load(open(f, "rb"))
            print(f"CFR baseline: {f} ({blob.get('iters', '?')} iterations, "
                  f"nash_conv {blob.get('nash_conv', float('nan')):.4f})")
            return blob["table"]
    raise SystemExit(f"no CFR table found; looked for {CFR_CANDIDATES}")


def setup():
    T = Tree()
    base = load_base()
    b = [T.from_table(base, q) for q in range(3)]
    return T, base, b


def strategy(T, base, b, name):
    """Our seat-0 strategy by name: CFR, 'RNR p=<p>', or BR(FF)."""
    if name == "CFR":
        return b[0]
    if name.startswith("RNR p="):
        p = name.split("=")[1]
        f = os.path.join(RESULTS, f"rnr_FF_a0.35_s0_independent_selfish_p{p}_strategy.npy")
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
        return np.load(f)
    if name == "BR(FF)":
        f1 = T.from_table(perturb(base, "F", 0.35), 1)
        f2 = T.from_table(perturb(base, "F", 0.35), 2)
        return T.respond([None, f1, f2], 0, T.util[:, 0], T.util[:, 0])[0]
    raise SystemExit(f"unknown strategy {name!r}; use CFR, 'RNR p=0.5', or BR(FF)")


def misspec():
    T, base, b = setup()
    eq = T.ev(b)[0]
    names = ["CFR", "RNR p=0.25", "RNR p=0.5", "RNR p=0.75", "RNR p=0.9", "BR(FF)"]
    S = {n: strategy(T, base, b, n) for n in names}
    opps = {}
    for a in (0.1, 0.2, 0.35, 0.5):
        opps[f"FF a{a}"] = (T.from_table(perturb(base, "F", a), 1), T.from_table(perturb(base, "F", a), 2))
    for t in "RCO":
        opps[f"{t}{t} a0.35"] = (T.from_table(perturb(base, t, .35), 1) if t != "O" else b[1],
                                 T.from_table(perturb(base, t, .35), 2) if t != "O" else b[2])
    print("\nour gain over equilibrium, strategy tuned for FF alpha 0.35, facing:")
    print(f"{'':<12}" + "".join(f"{k:>11}" for k in opps))
    for n, s in S.items():
        print(f"{n:<12}" + "".join(f"{T.ev([s, *o])[0] - eq:>+11.3f}" for o in opps.values()))


def own_reach_sa(T, sig_q, q):
    pe = np.ones(T.N)
    idx = T.kmask[q]
    pe[idx] = sig_q[T.esa[idx]]
    r = T.reach(pe)
    out = np.zeros(T.nSA[q])
    out[T.esa[idx]] = r[T.parent[idx]]
    return out


def random_strategy(T, q, rng):
    x = rng.random(T.nSA[q]) + 1e-3
    return T.normalize(x, q)


def bothadapt(names, iters, seed, report_every):
    T, base, b = setup()
    U = T.util
    eqv = T.ev(b); eq = eqv[0]
    rng = np.random.default_rng(seed) if seed is not None else None
    for n in names:
        s0 = strategy(T, base, b, n)
        sig, R, W = {}, {}, {}
        for q in (1, 2):
            # Random start = the strategy played on the FIRST iteration only. Regrets always
            # start at zero: seeding them with the strategy (an earlier version did) acts as
            # a prior worth thousands of iterations, because real regret updates are ~1e-4.
            sig[q] = random_strategy(T, q, rng) if rng is not None else T.uniform(q)
            R[q] = np.zeros(T.nSA[q])
            W[q] = np.zeros(T.nSA[q])
        start = f"random start, seed {seed}" if rng is not None else "uniform start"
        print(f"\n{n}: opponents learning from a {start}")
        print(f"{'iter':>6}{'our EV vs eq':>14}{'seat1 vs eq':>13}{'seat2 vs eq':>13}"
              f"{'gap seat1':>11}{'gap seat2':>11}{'secs':>7}")
        t0 = time.time()
        for t in range(1, iters + 1):
            for q in (1, 2):
                c = T.cfv([s0, sig[1], sig[2]], q, U[:, q])
                ev = np.bincount(T.sai[q], weights=sig[q] * c, minlength=T.nI[q])[T.sai[q]]
                R[q] = np.maximum(R[q] + c - ev, 0)
                sig[q] = T.rm(R[q], q)
                W[q] += t * own_reach_sa(T, sig[q], q) * sig[q]
            if t % report_every == 0 or t == iters:
                avg = {q: T.normalize(W[q], q) for q in (1, 2)}
                v = T.ev([s0, avg[1], avg[2]])
                g1 = T.respond([s0, None, avg[2]], 1, U[:, 1], U[:, 1])[1] - v[1]
                g2 = T.respond([s0, avg[1], None], 2, U[:, 2], U[:, 2])[1] - v[2]
                print(f"{t:>6}{v[0] - eq:>+14.3f}{v[1] - eqv[1]:>+13.3f}{v[2] - eqv[2]:>+13.3f}"
                      f"{g1:>11.4f}{g2:>11.4f}{time.time() - t0:>7.0f}",
                      flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["misspec", "bothadapt"])
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--strategies", nargs="+", default=["CFR"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--report-every", type=int, default=100)
    a = ap.parse_args()
    if a.mode == "misspec":
        misspec()
    else:
        bothadapt(a.strategies, a.iters, a.seed, a.report_every)