"""
The table chase, phase A: where does play go when every player models every other one correctly?

Each seat has a TABLE of strategies: the CFR equilibrium, the three starting styles (F, R, C at
alpha 0.3) and its entire bank of restricted responses (every solved p, every cell, pessimistic
bank). Players only ever play table strategies, and each knows exactly what the others play, so
there is no identification here (that is phase B). We run, exactly on the game tree:

  best-response dynamics  each player switches to its best table strategy against what the others
                          play now -- one player at a time ("sequential") or all at once
                          ("simultaneous") -- until nobody wants to switch (a pure equilibrium of
                          the table game) or a profile repeats (a cycle)
  fictitious play         each round every player best-responds to the others' AVERAGE play so far;
                          we track the average profile's payoffs and how far it is from an
                          equilibrium of the table game (the largest gain any player could get by
                          switching to a table strategy against it)

Speed comes from the sequence form. Expected payoff is linear in a player's realization plan (the
probability of each of its own action sequences), so one pass over the tree against the other two
players gives a payoff weight for each of our sequences, and every table strategy is then scored by
a dot product. Averages of play are averages of realization plans, so fictitious play is exact too.

    python table_chase.py                       # all seats' tables from the bank (pessimistic p*)
    python table_chase.py --random-starts 50    # more starting profiles for best-response dynamics
    python table_chase.py --standin             # testing without a bank: tilts stand in for it
Payoffs are chips/hand relative to each seat's equilibrium value.
"""

import argparse
import json
import pickle
import time

import numpy as np

from cfr_path import find_cfr
from session_sim import Strategies
from tree_engine import Tree

STYLES = [("F", 0.3), ("R", 0.3), ("C", 0.3)]      # the chase's starting styles, seats 0, 1, 2
EPS = 1e-9                                          # switch only for a strictly better strategy


class SeqForm:
    """Sequence-form view of one seat: realization plans and payoff weights per sequence."""

    def __init__(self, T, q):
        self.T, self.q = T, q
        last = np.full(T.N, -1, dtype=np.int64)          # seat q's last action on the path to each node
        for d in range(1, T.maxd + 1):
            nd = T.by_depth[d]
            last[nd] = last[T.parent[nd]]
            own = nd[T.kind[nd] == q]
            last[own] = T.esa[own]
        self.last = last
        e = T.kmask[q]
        par_i = np.full(T.nI[q], -1, dtype=np.int64)     # the sequence leading to each infoset
        par_i[T.sai[q][T.esa[e]]] = last[T.parent[e]]   # (perfect recall: the same at all its nodes)
        self.par_sa = par_i[T.sai[q]]
        depth = np.zeros(T.nSA[q], dtype=np.int64)       # longest chain of own actions, for iteration
        for _ in range(64):
            new = 1 + np.where(self.par_sa >= 0, depth[np.maximum(self.par_sa, 0)], 0)
            if np.array_equal(new, depth):
                break
            depth = new
        self.L = int(depth.max())

    def realization(self, S):
        """Behaviour strategies [K, nSA] -> realization plans [K, nSA]."""
        R = np.ones_like(S, dtype=np.float64)
        for _ in range(self.L):
            ext = np.concatenate([R, np.ones((R.shape[0], 1))], axis=1)    # column -1 = empty sequence
            R = S * ext[:, self.par_sa]
        return R

    def behaviour(self, R):
        """A realization plan (e.g. an average of several) -> the behaviour strategy that plays it."""
        T, q = self.T, self.q
        tot = np.bincount(T.sai[q], weights=R, minlength=T.nI[q])[T.sai[q]]
        uni = 1.0 / T.nact[q][T.sai[q]]
        return np.where(tot > 0, R / np.where(tot > 0, tot, 1.0), uni)

    def weights(self, sig):
        """(payoff with no own action taken, payoff weight per own sequence) against sig's others."""
        T = self.T
        w = T.reach(T.edge_probs(sig, skip=self.q)) * T.util[:, self.q]
        v = np.bincount(self.last + 1, weights=w, minlength=T.nSA[self.q] + 1)
        return v[0], v[1:]


def build_tables(T, S, standin=False):
    tables = []
    for q in range(3):
        names, flats = ["eq"], [S.base[q]]
        for t, a in STYLES:
            names.append(f"style {t}{a:g}"); flats.append(S.tilt(q, t, a))
        if standin:
            for t in "RCF":
                for a in (0.1, 0.2, 0.4, 0.5, 0.6):
                    names.append(f"tilt {t}{a:g}"); flats.append(S.tilt(q, t, a))
        else:
            import adapt_policies as ap
            seen = set()
            for cell, b in sorted(ap.load_bank(pess=True, seat=q).items()):
                for p in sorted(b["paths"]):
                    path = b["paths"][p]
                    if path in seen:
                        continue
                    seen.add(path)
                    names.append("p0" if p == 0 else f"{ap.cell_str(cell)}|{p:g}")
                    flats.append(np.load(path))
        tables.append(dict(names=names, S=np.array(flats, dtype=np.float64)))
    return tables


class Chase:
    def __init__(self, T, tables, eq):
        self.T, self.tab, self.eq = T, tables, eq
        self.sf = [SeqForm(T, q) for q in range(3)]
        for q in range(3):
            self.tab[q]["R"] = self.sf[q].realization(self.tab[q]["S"])

    def values(self, q, sig):
        """Payoff (vs equilibrium) of every table strategy of seat q against sig's other two."""
        v0, v = self.sf[q].weights(sig)
        return v0 + self.tab[q]["R"] @ v - self.eq[q], (v0, v)

    def sig_of(self, prof):
        return [self.tab[q]["S"][prof[q]] for q in range(3)]

    def br_dynamics(self, start, mode, max_steps=300):
        prof = list(start)
        path = []
        if mode == "simultaneous":
            seen = {tuple(prof): 0}
            for step in range(max_steps):
                vals = [self.values(q, self.sig_of(prof))[0] for q in range(3)]
                path.append(dict(profile=list(prof), payoff=[float(vals[q][prof[q]]) for q in range(3)]))
                new = [int(np.argmax(vals[q])) if vals[q].max() > vals[q][prof[q]] + EPS else prof[q] for q in range(3)]
                if new == prof:
                    return dict(outcome="fixed point", steps=step, path=path)
                prof = new
                if tuple(prof) in seen:
                    return dict(outcome="cycle", steps=step + 1, cycle_length=step + 1 - seen[tuple(prof)], path=path)
                seen[tuple(prof)] = step + 1
            return dict(outcome="no verdict", steps=max_steps, path=path)
        seen, still, q, step = {}, 0, 0, 0
        ev = self.T.ev(self.sig_of(prof)) - self.eq                     # the starting profile, recorded
        path.append(dict(profile=list(prof), mover=None, payoff=[float(x) for x in ev]))
        while step < max_steps and still < 3:
            state = (tuple(prof), q)
            if state in seen:
                return dict(outcome="cycle", steps=step, cycle_length=step - seen[state], path=path)
            seen[state] = step
            vals = self.values(q, self.sig_of(prof))[0]
            if vals.max() > vals[prof[q]] + EPS:
                prof[q] = int(np.argmax(vals)); still = 0
                ev = self.T.ev(self.sig_of(prof)) - self.eq
                path.append(dict(profile=list(prof), mover=q, payoff=[float(x) for x in ev]))
            else:
                still += 1
            q = (q + 1) % 3; step += 1
        return dict(outcome="fixed point" if still >= 3 else "no verdict", steps=step, path=path)

    def fictitious_play(self, start, rounds=300, every=10):
        Rsum = [self.tab[q]["R"][start[q]].copy() for q in range(3)]
        counts = [np.zeros(len(self.tab[q]["names"])) for q in range(3)]
        for q in range(3):
            counts[q][start[q]] += 1
        trace = []
        for t in range(1, rounds + 1):
            avg = [self.sf[q].behaviour(Rsum[q] / t) for q in range(3)]
            picks, row = [], dict(round=t, payoff=[], gap=[])
            for q in range(3):
                vals, (v0, v) = self.values(q, avg)
                own = v0 + (Rsum[q] / t) @ v - self.eq[q]           # the average strategy's payoff
                row["payoff"].append(float(own)); row["gap"].append(float(vals.max() - own))
                picks.append(int(np.argmax(vals)))
            for q in range(3):
                Rsum[q] += self.tab[q]["R"][picks[q]]; counts[q][picks[q]] += 1
            if t % every == 0 or t == rounds:
                trace.append(row)
        top = [[(self.tab[q]["names"][k], float(counts[q][k] / counts[q].sum()))
                for k in np.argsort(-counts[q])[:5]] for q in range(3)]
        return dict(trace=trace, top=top)


def describe(ch, res):
    p = res["path"][-1]["profile"] if res["path"] else None
    names = [ch.tab[q]["names"][p[q]] for q in range(3)] if p else []
    return names


def attractor(res, mode):
    """The profiles a run ends on: [fixed point] or the profiles of its final loop, with payoffs."""
    path = res["path"]
    if not path:                                    # saved by an older version that recorded no moves
        return []
    if res["outcome"] == "fixed point":
        return [path[-1]]
    if res["outcome"] != "cycle":
        return path[-3:]
    if mode == "simultaneous":                      # one entry per round: the loop is the last L entries
        return path[-res["cycle_length"]:]
    last = path[-1]["profile"]                      # one entry per switch: back to the last profile
    prev = max(i for i, e in enumerate(path[:-1]) if e["profile"] == last) if any(
        e["profile"] == last for e in path[:-1]) else 0
    return path[prev + 1:]


def report(path):
    """Group the saved runs by where they end up: which fixed point or which loop, how often, and the
    payoffs over it (each seat's lowest, highest and average over the loop's profiles)."""
    D = json.load(open(path)); names = D["tables"]
    for mode in ("sequential", "simultaneous"):
        groups = {}
        for key, res in D["dynamics"].items():
            start, m = key.split("|")
            if m != mode:
                continue
            att = attractor(res, mode)
            if not att:
                continue
            sig = frozenset(tuple(e["profile"]) for e in att)
            g = groups.setdefault(sig, dict(starts=[], att=att, outcome=res["outcome"]))
            g["starts"].append(start)
        print(f"\n{mode}: {len(groups)} distinct end states")
        for g in sorted(groups.values(), key=lambda g: -len(g["starts"])):
            pay = np.array([e["payoff"] for e in g["att"]])
            kind = "fixed point" if g["outcome"] == "fixed point" else f"loop of {len(g['att'])} profiles"
            print(f"  {kind}, reached from {len(g['starts'])} start(s): "
                  + ", ".join(g["starts"][:4]) + (" ..." if len(g["starts"]) > 4 else ""))
            for q in range(3):
                print(f"    seat {q}: payoff {pay[:, q].min():+.3f} to {pay[:, q].max():+.3f} "
                      f"(average {pay[:, q].mean():+.3f})")
            shown = g["att"] if len(g["att"]) <= 6 else g["att"][:6]
            for e in shown:
                print("      " + " / ".join(names[q][e["profile"][q]] for q in range(3)))
            if len(g["att"]) > 6:
                print(f"      ... and {len(g['att']) - 6} more")


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--standin", action="store_true", help="tilts instead of the bank (testing)")
    ap_.add_argument("--random-starts", type=int, default=20)
    ap_.add_argument("--rounds", type=int, default=300, help="fictitious play rounds")
    ap_.add_argument("--seed", type=int, default=0)
    ap_.add_argument("--out", default="table_chase.json")
    ap_.add_argument("--report", action="store_true", help="group the saved runs by where they end up")
    a = ap_.parse_args()
    if a.report:
        return report(a.out)
    t0 = time.time()
    T = Tree()
    base = pickle.load(open(find_cfr(), "rb"))["table"]
    S = Strategies(T, base)
    eq = T.ev(S.base)
    tables = build_tables(T, S, a.standin)
    ch = Chase(T, tables, eq)
    print(f"tables: {[len(t['names']) for t in tables]} strategies per seat  [{time.time() - t0:.0f}s]", flush=True)
    rng = np.random.default_rng(a.seed)
    starts = {"styles": [1, 2, 3], "equilibrium": [0, 0, 0]}
    for i in range(a.random_starts):
        starts[f"random {i}"] = [int(rng.integers(len(tables[q]["names"]))) for q in range(3)]
    out = dict(tables=[t["names"] for t in tables], dynamics={}, fictitious={})
    print(f"\nbest-response dynamics (payoffs vs equilibrium at the end)")
    print(f"{'start':<14}{'mode':<14}{'outcome':<14}{'steps':>6}{'cycle':>7}   final payoffs           final profile")
    tally = {}
    for name, st in starts.items():
        for mode in ("sequential", "simultaneous"):
            r = ch.br_dynamics(st, mode)
            out["dynamics"][f"{name}|{mode}"] = r
            tally.setdefault(mode, []).append(r["outcome"])
            if name in ("styles", "equilibrium") or r["outcome"] != "fixed point" or name == "random 0":
                pay = r["path"][-1]["payoff"] if r["path"] else [0.0, 0.0, 0.0]
                print(f"{name:<14}{mode:<14}{r['outcome']:<14}{r['steps']:>6}{r.get('cycle_length', ''):>7}   "
                      f"{' '.join(f'{x:+.3f}' for x in pay):<24}{' / '.join(describe(ch, r))}", flush=True)
    for mode, v in tally.items():
        print(f"  {mode}: " + ", ".join(f"{v.count(o)} {o}" for o in ("fixed point", "cycle", "no verdict")))
    for name in ("styles", "equilibrium"):
        f = ch.fictitious_play(starts[name], a.rounds)
        out["fictitious"][name] = f
        last = f["trace"][-1]
        print(f"\nfictitious play from {name}, {a.rounds} rounds: average-profile payoffs "
              f"{' '.join(f'{x:+.3f}' for x in last['payoff'])}; largest gain from switching "
              f"{' '.join(f'{x:.4f}' for x in last['gap'])}")
        for q in range(3):
            print(f"  seat {q} plays most: " + ", ".join(f"{n} {w:.0%}" for n, w in f["top"][q]))
    json.dump(out, open(a.out, "w"))
    print(f"\nsaved {a.out}  [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()