"""
The table chase, phase B: the realistic version of phase A (table_chase.py).

Three players, each with its seat's TABLE of strategies (equilibrium, the three styles, its whole bank).
Nobody sees what the others play. Each player keeps an exact Bayesian posterior over which table
strategy each opponent is playing (a TableIdentifier), and at its update hands switches to its best
table strategy against the PREDICTED opponents: the posterior-weighted average of their table
strategies (exact in the sequence form, see table_chase.py).

The question: phase A showed that best-responding to what opponents play NOW cycles when everyone moves
together, while best-responding to their AVERAGE play (fictitious play) settles near equilibrium. A
long memory makes beliefs an average of past play; a short one tracks current play. So:

  memory   none (lam = 1), ~50 hands (0.98), ~20 hands (0.95)
  timing   sync     all players update at the same hands (phase A's "simultaneous")
           stagger  seat q updates q/3 of an interval later than seat 0 (phase A's "sequential")

Everything starts from the styles (seat 0 over-folder, 1 over-raiser, 2 over-caller, all at 0.3) and
plays each style until the first update. Every interval we record the profile, each seat's exact payoff
(vs equilibrium), how much posterior each player puts on what its opponents really play, and how far
its predicted payoff was from its actual one.

    python table_chase_b.py                      # 3 memories x 2 timings x 5 sessions of 4,000 hands
    python table_chase_b.py --report
    python table_chase_b.py --standin --hands 600 --sessions 1      # quick test without a bank
Memory: each worker holds all three tables (~0.3 GB); --workers defaults to 4.
"""

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

OUT = "chaseB"
START = [1, 2, 3]                                   # table indices of the styles F, R, C (seats 0, 1, 2)
G = {}


# ---------------------------------------------------------------- identifier

class TableIdentifier:
    """Exact posterior over which table strategy each of `seat`'s two opponents is playing.

    Uniform prior; cards are revealed only at showdown (the realistic regime). tremble mixes a little
    uniform play into every likelihood so that one unlikely action cannot rule a strategy out forever.
    forget < 1 discounts old evidence: memory ~ 1 / (1 - forget) hands."""

    def __init__(self, game, T, S, seat, forget=1.0, tremble=0.01):
        self.game, self.T, self.seat, self.forget, self.tremble = game, T, seat, forget, tremble
        self.opps = [o for o in range(3) if o != seat]
        self.S = {j: S[j] for j in self.opps}
        self.loglik = np.zeros((len(S[self.opps[0]]), len(S[self.opps[1]])))
        self.deck = [a for a, _ in game.new_initial_state().chance_outcomes()]
        self.keyidx = {j: {k: i for i, k in enumerate(T.keys[j])} for j in self.opps}
        self.sa_cache = {}

    def _sa(self, j, key, a):
        c = self.sa_cache.get((j, key, a))
        if c is None:
            i = self.keyidx[j][key]
            off, n = int(self.T.off[j][i]), int(self.T.nact[j][i])
            acts = list(self.T.saa[j][off:off + n])
            c = (off + acts.index(a), n)
            self.sa_cache[(j, key, a)] = c
        return c

    def _likelihoods(self, history):
        """{opponent: [K x C] likelihood of its actions per table strategy, per card it might hold}."""
        g = self.game
        c0, board, st = history[self.seat], None, g.new_initial_state()
        for i, a in enumerate(history):
            if i >= 3 and st.is_chance_node():
                board = a
            st.apply_action(a)
        free = [c for c in self.deck if c != c0 and c != board]
        out = {}
        for j in self.opps:
            other = [o for o in self.opps if o != j][0]
            cols = []
            for c in free:
                h = list(history); h[j] = c
                h[other] = next(x for x in free if x != c)       # any legal card: j's view ignores it
                st = g.new_initial_state(); L = np.ones(len(self.S[j]))
                for a in h:
                    if not st.is_chance_node() and st.current_player() == j:
                        sa, n = self._sa(j, st.information_state_string(j), a)
                        L *= (1 - self.tremble) * self.S[j][:, sa] + self.tremble / n
                    st.apply_action(a)
                cols.append(L)
            out[j] = np.stack(cols, axis=1)
        return out, free

    def update(self, history, true_cards, showdown_seats):
        L, free = self._likelihoods(history)
        o1, o2 = self.opps
        C = len(free)
        M = np.ones((C, C)) - np.eye(C)                          # the two opponents hold different cards
        for j, axis in ((o1, 0), (o2, 1)):
            if j in showdown_seats:                              # a revealed card rules other deals out
                keep = np.array([c == true_cards[j] for c in free], float)
                M = M * (keep[:, None] if axis == 0 else keep[None, :])
        like = L[o1] @ M @ L[o2].T
        with np.errstate(divide="ignore"):
            self.loglik = self.forget * self.loglik + np.log(np.maximum(like, 1e-300))

    def marginals(self):
        P = np.exp(self.loglik - self.loglik.max()); P /= P.sum()
        return {self.opps[0]: P.sum(1), self.opps[1]: P.sum(0)}


# ---------------------------------------------------------------- session

def init(standin, tables_cache=None):
    import pyspiel
    from cfr_path import find_cfr
    from session_sim import Strategies
    from tree_engine import Tree
    import table_chase as tc
    T = Tree()
    base = pickle.load(open(find_cfr(), "rb"))["table"]
    S = Strategies(T, base)
    eq = T.ev(S.base)
    tables = tc.build_tables(T, S, standin)
    ch = tc.Chase(T, tables, eq)
    for q in range(3):                  # single precision is plenty for choosing strategies; halves memory
        tables[q]["S"] = tables[q]["S"].astype(np.float32)
        tables[q]["R"] = tables[q]["R"].astype(np.float32)
        tables[q]["S32"] = tables[q]["S"]
    G.update(game=pyspiel.load_game("leduc_poker", {"players": 3}), T=T, S=S, eq=eq, tab=tables, ch=ch,
             policy_cache={})


def policy(q, k):
    c = G["policy_cache"]
    if (q, k) not in c:                 # back to double precision, each decision renormalised to sum to 1
        T = G["T"]; f = G["tab"][q]["S"][k].astype(np.float64)
        f /= np.bincount(T.sai[q], weights=f, minlength=T.nI[q])[T.sai[q]]
        c[(q, k)] = G["S"].table(q, f)
    return c[(q, k)]


def predicted(q, marg):
    """Seat q's prediction of the others: posterior-weighted average of their table strategies."""
    sig = [None] * 3
    for j, m in marg.items():
        R = m @ G["tab"][j]["R"]
        sig[j] = G["ch"].sf[j].behaviour(R)
    sig[q] = G["tab"][q]["S"][0]
    return sig


def run(job):
    lam, timing, seed, hands, every = job
    from q2_identify import play_hand
    T, ch, tab = G["T"], G["ch"], G["tab"]
    rng = np.random.default_rng([seed, int(lam * 1000), 0 if timing == "sync" else 1])
    ids = [TableIdentifier(G["game"], T, [tab[j]["S32"] for j in range(3)], q, forget=lam) for q in range(3)]
    prof = list(START)
    pred_val = [None] * 3
    offset = [0, 0, 0] if timing == "sync" else [0, every // 3, 2 * every // 3]
    log, t0 = [], time.time()
    for n in range(1, hands + 1):
        hist, cards, sd, _ = play_hand(G["game"], [policy(q, prof[q]) for q in range(3)], rng)
        for q in range(3):
            ids[q].update(hist, cards, sd)
        for q in range(3):
            if n >= every and (n - offset[q]) % every == 0 and n < hands:
                vals = ch.values(q, predicted(q, ids[q].marginals()))[0]
                k = int(np.argmax(vals)) if vals.max() > vals[prof[q]] + 1e-9 else prof[q]
                prof[q], pred_val[q] = k, float(vals[k])
        if n % every == 0:
            sig = ch.sig_of(prof)
            actual = [float(ch.values(q, sig)[0][prof[q]]) for q in range(3)]
            mass = []
            for q in range(3):
                m = ids[q].marginals()
                mass.append(float(np.mean([m[j][prof[j]] for j in m])))
            log.append(dict(hand=n, profile=list(prof), payoff=actual, mass=mass,
                            pred_err=[None if pred_val[q] is None else abs(pred_val[q] - actual[q]) for q in range(3)]))
    return job, dict(lam=lam, timing=timing, seed=seed, hands=hands, every=every, log=log,
                     names=[[tab[q]["names"][k] for k in sorted({e["profile"][q] for e in log})] for q in range(3)],
                     secs=time.time() - t0)


def sim_file(job):
    lam, timing, seed, hands, every = job
    return os.path.join(OUT, f"chase_l{lam:g}_{timing}_s{seed}_h{hands}_e{every}.json")


# ---------------------------------------------------------------- report

def classify(log, tail=40):
    """Over the second half of the session (at most its last `tail` records): settled (one profile),
    a cycle of period P, or drift."""
    seq = [tuple(e["profile"]) for e in log][len(log) // 2:][-tail:]
    if len(set(seq)) == 1:
        return "settled", 1
    for P in range(2, len(seq) // 2 + 1):
        if all(seq[i] == seq[i + P] for i in range(len(seq) - P)):
            return "cycle", P
    return "drift", None


def report(a):
    rows = {}
    for f in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        r = json.load(open(os.path.join(OUT, f)))
        rows.setdefault((r["hands"], r["every"], r["lam"], r["timing"]), []).append(r)
    if not rows:
        raise SystemExit("no sessions yet")
    print(f"{'hands':>6}{'every':>6}{'memory':>8}{'timing':>9}{'n':>4}   outcome (second half)            "
          f"{'switches/1k':>12}{'payoff swing':>14}{'average payoffs (0/1/2)':>27}{'mass on truth':>15}{'pred. error':>13}")
    for (hands, every, lam, timing), R in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1], -kv[0][2], kv[0][3])):
        outs = [classify(r["log"]) for r in R]
        tally = ", ".join(f"{sum(o[0] == k for o in outs)} {k}" for k in ("settled", "cycle", "drift")
                          if any(o[0] == k for o in outs))
        periods = sorted({o[1] for o in outs if o[0] == "cycle"})
        if periods:
            tally += f" (periods {periods})"
        half = [r["log"][len(r["log"]) // 2:] for r in R]
        sw = [sum(a_["profile"] != b["profile"] for a_, b in zip(h[:-1], h[1:])) / (len(h) * r["every"]) * 1000
              for h, r in zip(half, R)]
        swing = [max(np.ptp([e["payoff"][q] for e in h]) for q in range(3)) for h in half]
        avg = np.mean([[np.mean([e["payoff"][q] for e in h]) for q in range(3)] for h in half], axis=0)
        mass = [np.mean([np.mean(e["mass"]) for e in h]) for h in half]
        err = [np.mean([x for e in h for x in e["pred_err"] if x is not None]) for h in half]
        mem = "none" if lam == 1.0 else f"~{1 / (1 - lam):.0f}"
        print(f"{hands:>6}{every:>6}{mem:>8}{timing:>9}{len(R):>4}   {tally:<34}{np.median(sw):>12.1f}{np.median(swing):>14.2f}"
              f"{'  '.join(f'{x:+.3f}' for x in avg):>27}{np.median(mass):>15.0%}{np.median(err):>13.3f}")
    print("\nAll over the second half of each session. switches/1k: profile changes per 1,000 hands. payoff "
          "swing: the largest range of any seat's payoff (chips/hand vs equilibrium). mass on truth: average "
          "posterior each player puts on the strategies its opponents really play. pred. error: |predicted - "
          "actual| payoff of each player's own choice.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lams", type=float, nargs="+", default=[1.0, 0.98, 0.95])
    p.add_argument("--timings", nargs="+", default=["sync", "stagger"])
    p.add_argument("--sessions", type=int, default=5); p.add_argument("--hands", type=int, default=4000)
    p.add_argument("--every", type=int, default=50); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--standin", action="store_true"); p.add_argument("--report", action="store_true")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not a.report:
        jobs = [(lam, tm, sd, a.hands, a.every) for lam in a.lams for tm in a.timings for sd in range(a.sessions)]
        todo = [j for j in jobs if not os.path.exists(sim_file(j))]
        print(f"{len(todo)} of {len(jobs)} sessions to run on {a.workers} workers", flush=True)
        t0 = time.time()
        with Pool(a.workers, initializer=init, initargs=(a.standin,)) as pool:
            for i, (job, r) in enumerate(pool.imap_unordered(run, todo), 1):
                json.dump(r, open(sim_file(job), "w"))
                print(f"[{(time.time() - t0) / 60:6.1f} min] {i}/{len(todo)}  memory {job[0]:g}, {job[1]}, "
                      f"session {job[2]} ({r['secs'] / 60:.1f} min)", flush=True)
    report(a)


if __name__ == "__main__":
    main()