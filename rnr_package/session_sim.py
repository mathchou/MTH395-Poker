"""
Session simulator: three players at a table, any of whom may adapt.

Each seat is an AGENT:

  fixed     plays one style throughout: equilibrium (O) or a tilt TYPE:ALPHA (e.g. F:0.35);
            optionally switches to another style at a given hand (e.g. F:0.35>R:0.3@200)
  adaptive  runs its own grid identifier on the other two players and, every --every hands,
            recomputes its strategy:
               bbr    (1 - w) * base + w * best response to the posterior's prediction of the
                      opponents, with w = c ** beta, where c is the posterior's confidence
               bank   the restricted response from the solved bank nearest to the posterior's
                      (types, alphas), at p = p_max * c ** beta (falls back to bbr if the bank
                      has nothing for those types)
            base is the agent's own style when it is not adapting: equilibrium by default,
            or a tilt ("an over-folder who learns")

Confidence c = (P(most likely type pair) - 1/16) / (1 - 1/16): 0 at the prior, 1 when certain.
The posterior's prediction of an opponent mixes its hypotheses' strategies, weighted by the
posterior and by how likely each hypothesis is to reach each spot. The two opponents are
predicted independently (the posterior's correlation between them is ignored).

Recorded at every recompute (and at the start): each seat's EXACT expected payoff for the
current profile (no card-luck noise), each adaptive agent's confidence, most likely types and
alpha estimates, and how much its strategy changed (reach-weighted total variation).
Realised chips per hand are recorded too.

    python session_sim.py --seat0 adaptive --seat1 F:0.35 --seat2 R:0.2 --hands 400
    python session_sim.py --seat0 adaptive --seat1 adaptive --seat2 R:0.3 --hands 400
    python session_sim.py --seat0 adaptive:F:0.3 --seat1 adaptive:R:0.3 --seat2 adaptive:C:0.3
    python session_sim.py ... --mode bank --beta 2 --every 25 --sessions 5 --out run.json
    python session_sim.py --plot run.json
    python session_sim.py ... --forget 0.98       # forget old evidence (memory ~ 50 hands)
    python session_sim.py ... --switch 0.02       # or: allow each opponent to change type
"""

import argparse
import glob
import json
import os
import pickle
import re
import time

import numpy as np
import pyspiel

from cfr_path import find_cfr
from q2_grid_identify import GridIdentifier, TYPES as TILTS
from q2_identify import FAV, play_hand
from rnr_general import own_reach_sa
from tree_engine import Tree

MIN_W = 1e-5            # hypotheses with less posterior weight are left out of predictions


# ------------------------------------------------------------------ strategies

class Strategies:
    """Flat strategy arrays (tree_engine format) and conversion to playable tables."""

    def __init__(self, T, base_table):
        self.T, self.base_table = T, base_table
        self.base = [T.from_table(base_table, q) for q in range(3)]
        self.isum = [np.bincount(T.sai[q], weights=self.base[q], minlength=T.nI[q]) for q in range(3)]

    def tilt(self, q, t, a):
        """Seat q's equilibrium strategy tilted toward type t's action by alpha a."""
        T = self.T
        if t == "O" or a == 0:
            return self.base[q]
        fav = T.saa[q] == FAV[t]
        legal = np.bincount(T.sai[q], weights=fav.astype(float), minlength=T.nI[q])[T.sai[q]] > 0
        tot = self.isum[q][T.sai[q]]
        tilted = ((1 - a) * self.base[q] + a * fav) / ((1 - a) * tot + a)
        return np.where(legal, tilted, self.base[q])

    def table(self, q, flat):
        """Playable {infostate: {action: prob}} for seat q."""
        T = self.T
        out = {}
        for i, key in enumerate(T.keys[q]):
            lo, hi = T.off[q][i], T.off[q][i + 1]
            out[key] = {int(T.saa[q][j]): float(flat[j]) for j in range(lo, hi)}
        return out

    def tv(self, q, a, b, reach_profile):
        """Total variation between two strategies of seat q, weighted by how often play reaches
        each information set under reach_profile (all three seats' strategies)."""
        T = self.T
        idx = T.kmask[q]
        r = T.reach(T.edge_probs(reach_profile, skip=q))
        w = np.zeros(T.nSA[q]); w[T.esa[idx]] = r[T.parent[idx]]
        wI = np.bincount(T.sai[q], weights=w, minlength=T.nI[q]) / T.nact[q]
        d = np.bincount(T.sai[q], weights=0.5 * np.abs(a - b), minlength=T.nI[q])
        return float((d * wI).sum() / max(wI.sum(), 1e-12))


# ------------------------------------------------------------------ the bank

def load_bank(seat):
    """Solved restricted responses for this seat: {(t1, a1, t2, a2): {p: path}}, from results/
    (equal-alpha runs) and results_het/ (independent alphas). Types with alpha 0 are O."""
    bank = {}
    pat = re.compile(r"rnr_([ORCF]{2})_a([0-9.]+)(?:-([0-9.]+))?_s(\d)_independent_selfish_p([0-9.]+)_strategy\.npy$")
    for f in glob.glob("results/rnr_*_strategy.npy") + glob.glob("results_het/rnr_*_strategy.npy"):
        m = pat.search(os.path.basename(f))
        if not m or int(m.group(4)) != seat:
            continue
        pair, a1 = m.group(1), float(m.group(2))
        a2 = float(m.group(3)) if m.group(3) else a1
        t1, t2 = pair
        a1, a2 = (a1 if t1 != "O" else 0.0), (a2 if t2 != "O" else 0.0)
        t1, t2 = (t1 if a1 > 0 else "O"), (t2 if a2 > 0 else "O")
        bank.setdefault((t1, a1, t2, a2), {})[float(m.group(5))] = f
    return bank


# ------------------------------------------------------------------ agents

class Fixed:
    def __init__(self, seat, spec, S):
        self.seat, self.S = seat, S
        self.phases = []                                   # [(from_hand, type, alpha)]
        for part in spec.split(">"):
            style, _, at = part.partition("@")
            t, _, a = style.partition(":")
            self.phases.append((int(at) if at else 0, t, float(a or 0)))
        self.flat = None; self.update(0)

    def update(self, hand):
        t, a = [(t, a) for start, t, a in self.phases if start <= hand][-1]
        new = self.S.tilt(self.seat, t, a)
        changed = self.flat is None or not np.array_equal(new, self.flat)
        self.flat, self.tab = new, (self.S.table(self.seat, new) if changed else self.tab)
        self.label = f"{t}:{a:g}"

    def observe(self, *a):
        pass


class Adaptive:
    def __init__(self, seat, base_spec, S, game, args, bank):
        self.seat, self.S, self.args, self.bank = seat, S, args, bank
        self.opps = [o for o in range(3) if o != seat]
        t, _, a = base_spec.partition(":")
        self.base_flat = S.tilt(seat, t or "O", float(a or 0))
        self.flat, self.tab = self.base_flat, S.table(seat, self.base_flat)
        self.gi = GridIdentifier(game, S.base_table, seat=seat, tremble=args.tremble,
                                 forget=getattr(args, "forget", 1.0), switch=getattr(args, "switch", 0.0))
        self.label = f"adaptive({base_spec or 'O'})"
        self.conf, self.map_types, self.alphas, self.used = 0.0, ("O", "O"), (0.0, 0.0), "base"

    def observe(self, hist, cards, showdown):
        self.gi.update(hist, cards, showdown)

    def _predict(self, M, o):
        """Posterior-predictive strategy of opponent seat o from its marginal M over hypotheses."""
        S, T = self.S, self.S.T
        num = np.zeros(T.nSA[o]); den = np.zeros(T.nSA[o])
        for h, w in enumerate(M):
            if w < MIN_W:
                continue
            t, a = self.gi.hyp[h]
            sig = S.tilt(o, t, a)
            r = own_reach_sa(T, sig, o)
            num += w * r * sig; den += w * r
        return np.where(den > 0, num / np.where(den > 0, den, 1), S.base[o])

    def recompute(self):
        P, summ = self.gi.summary(self.args.regime)
        o1, o2 = self.opps
        pt = max((self.gi.p_types(P, x, y), x, y) for x in ["O"] + list(TILTS) for y in ["O"] + list(TILTS))
        # clamp: rounding can push a near-certain posterior's total a hair above 1, which would
        # make w > 1 and give the base style a (tiny) negative weight
        self.conf = min(1.0, max(0.0, (pt[0] - 1 / 16) / (1 - 1 / 16)))
        self.map_types = (pt[1], pt[2])
        self.alphas = (summ[o1]["alpha_mean"], summ[o2]["alpha_mean"])
        w = self.conf ** self.args.beta
        T = self.S.T
        if self.args.mode == "bank":
            pick = self._from_bank(P)
            if pick is not None:
                self.flat, self.used = pick
                self.tab = self.S.table(self.seat, self.flat)
                return
        pred = {o1: self._predict(P.sum(1), o1), o2: self._predict(P.sum(0), o2)}
        prof = [None] * 3
        for o in self.opps:
            prof[o] = pred[o]
        br, _ = T.respond(prof, self.seat, T.util[:, self.seat], T.util[:, self.seat])
        self.flat = (1 - w) * self.base_flat + w * br
        self.used = f"bbr w={w:.2f}"
        self.tab = self.S.table(self.seat, self.flat)

    def _from_bank(self, P):
        """Nearest bank entry for the most likely types; p from confidence. None if no entry."""
        o1, o2 = self.opps
        t1, t2 = self.map_types
        def a_given(Mseat, t):                  # posterior-mean alpha given the type
            if t == "O":
                return 0.0
            idx = [i for i, h in enumerate(self.gi.hyp) if h[0] == t]
            m = Mseat[idx]
            return float(m @ np.array([self.gi.hyp[i][1] for i in idx]) / max(m.sum(), 1e-12))
        a1, a2 = a_given(P.sum(1), t1), a_given(P.sum(0), t2)
        cands = [k for k in self.bank if k[0] == t1 and k[2] == t2]
        if not cands:
            return None
        key = min(cands, key=lambda k: abs(k[1] - a1) + abs(k[3] - a2))
        p_target = self.args.p_max * self.conf ** self.args.beta
        ps = sorted(p for p in self.bank[key] if p <= p_target + 1e-9)
        if not ps:
            return None
        p = ps[-1]
        return np.load(self.bank[key][p]), f"bank {key[0]}{key[1]:g}/{key[2]}{key[3]:g} p={p:g}"


def make_agent(seat, spec, S, game, args, bank):
    if spec.startswith("adaptive"):
        return Adaptive(seat, spec.partition(":")[2], S, game, args, bank)
    return Fixed(seat, spec, S)


# ------------------------------------------------------------------ the session

def run_session(agents, S, game, args, rng):
    T = S.T
    trace, chips = [], []
    prev = [a.flat.copy() for a in agents]

    def snapshot(hand):
        prof = [a.flat for a in agents]
        ev = T.ev(prof)
        eq = T.ev(S.base)
        row = dict(hand=hand, ev=[float(ev[q] - eq[q]) for q in range(3)], agents=[])
        for q, a in enumerate(agents):
            info = dict(label=a.label)
            if isinstance(a, Adaptive):
                info.update(conf=a.conf, types="".join(a.map_types), alphas=list(a.alphas), used=a.used,
                            change=S.tv(q, prev[q], a.flat, prof))
            elif isinstance(a, Fixed):
                info.update(change=S.tv(q, prev[q], a.flat, prof))
            row["agents"].append(info)
        for q in range(3):
            prev[q] = agents[q].flat.copy()
        trace.append(row)

    snapshot(0)
    for n in range(1, args.hands + 1):
        hist, cards, showdown, returns = play_hand(game, [a.tab for a in agents], rng)
        chips.append([float(x) for x in returns])
        for a in agents:
            a.observe(hist, cards, showdown)
        changed = False
        for a in agents:
            if isinstance(a, Fixed):
                before = a.label; a.update(n); changed |= a.label != before
        if n % args.every == 0:
            for a in agents:
                if isinstance(a, Adaptive):
                    a.recompute()
            changed = True
        if changed or n == args.hands:
            snapshot(n)
    return dict(trace=trace, chips=chips)


def report(res, specs):
    tr = res["trace"]; chips = np.array(res["chips"])
    print(f"\n{'hand':>6} " + " ".join(f"{'seat ' + str(q) + ' EV':>11}" for q in range(3)) + "   adaptive agents")
    for row in tr:
        notes = []
        for q, a in enumerate(row["agents"]):
            if "conf" in a:
                notes.append(f"s{q}: c={a['conf']:.2f} {a['types']} "
                             f"a=({a['alphas'][0]:.2f},{a['alphas'][1]:.2f}) {a['used']} change {a['change']:.3f}")
        print(f"{row['hand']:>6} " + " ".join(f"{e:>+11.3f}" for e in row["ev"]) + "   " + " | ".join(notes))
    m = chips.mean(0); se = chips.std(0) / np.sqrt(len(chips))
    print("realised chips/hand (raw, with card luck): " +
          ", ".join(f"seat {q} {m[q]:+.3f} +/- {se[q]:.3f}" for q in range(3)))


def plot(path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = json.load(open(path))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    for s, res in enumerate(D["sessions"]):
        tr = res["trace"]; h = [r["hand"] for r in tr]
        for q in range(3):
            axs[0].plot(h, [r["ev"][q] for r in tr], color=f"C{q}", alpha=0.6, lw=1,
                        label=f"seat {q}: {D['specs'][q]}" if s == 0 else None)
            if "conf" in tr[-1]["agents"][q]:
                axs[1].plot(h, [r["agents"][q].get("conf", 0) for r in tr], color=f"C{q}", alpha=0.6, lw=1,
                            label=f"seat {q}" if s == 0 else None)
    axs[0].axhline(0, color="k", lw=0.6); axs[0].set_xlabel("hand"); axs[0].set_ylabel("exact EV vs equilibrium")
    axs[0].set_title("Expected payoff of the current profile"); axs[0].legend(fontsize=8)
    axs[1].set_xlabel("hand"); axs[1].set_ylabel("confidence c"); axs[1].set_title("Adaptive agents' confidence")
    axs[1].legend(fontsize=8)
    plt.tight_layout(); out = path.replace(".json", ".png"); plt.savefig(out, dpi=110)
    print("plot:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seat0", default="adaptive"); ap.add_argument("--seat1", default="F:0.35")
    ap.add_argument("--seat2", default="R:0.2")
    ap.add_argument("--hands", type=int, default=400); ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--every", type=int, default=25, help="recompute adaptive strategies every N hands")
    ap.add_argument("--mode", choices=["bbr", "bank"], default="bbr")
    ap.add_argument("--beta", type=float, default=1.0, help="w (or p / p_max) = confidence ** beta")
    ap.add_argument("--p-max", type=float, default=0.5, help="bank mode: largest p used")
    ap.add_argument("--tremble", type=float, default=0.01, help="identifier's random-play allowance")
    ap.add_argument("--forget", type=float, default=1.0,
                    help="exponential forgetting per hand (memory ~ 1/(1-forget) hands); 1 = none")
    ap.add_argument("--switch", type=float, default=0.0,
                    help="fixed-share: chance per hand that an opponent re-draws its type; 0 = none")
    ap.add_argument("--regime", choices=["actions", "showdown"], default="showdown",
                    help="what the identifier sees (showdown = cards shown at showdown)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None, help="plot a saved run and exit")
    a = ap.parse_args()
    if a.plot:
        return plot(a.plot)
    game = pyspiel.load_game("leduc_poker", {"players": 3})
    T = Tree(); S = Strategies(T, pickle.load(open(find_cfr(), "rb"))["table"])
    specs = [a.seat0, a.seat1, a.seat2]
    rng = np.random.default_rng(a.seed)
    out = dict(specs=specs, args=vars(a), sessions=[])
    for k in range(a.sessions):
        t0 = time.time()
        agents = [make_agent(q, specs[q], S, game, a, load_bank(q) if a.mode == "bank" else {}) for q in range(3)]
        res = run_session(agents, S, game, a, rng)
        print(f"\n=== session {k + 1}: " + "  ".join(f"seat {q} = {specs[q]}" for q in range(3)) +
              f"   [{time.time() - t0:.0f}s]")
        report(res, specs)
        out["sessions"].append(res)
    if a.out:
        json.dump(out, open(a.out, "w")); print("saved", a.out)


if __name__ == "__main__":
    main()
