"""
Bayesian opponent identification with an independent, (near-)continuous bias per opponent.

q2_identify.py assumes both opponents share one known alpha and chooses among 16 type pairs.
Here each opponent has its own type AND its own alpha, on a fine grid:

    per opponent:  "equilibrium" (alpha = 0)  +  R, C, F at alpha = 0.05, 0.10, ..., 0.60
                   = 37 hypotheses;  joint over the two opponents = 37 x 37 = 1,369
    prior:         P(equilibrium) = 0.25 per opponent, the rest spread evenly

At alpha = 0 every type IS the equilibrium player, so "O" is just the alpha = 0 point.

Why it's cheap: a tilted opponent plays action a at an information set with probability
    ((1 - alpha) * sigma*(a) + alpha * [a is the favoured action]) / normaliser
(unchanged if the favoured action is illegal there). So each possible deal is replayed ONCE,
and this formula is evaluated for all 37 hypotheses at once with numpy.

We sit in seat 0 and play the equilibrium, as in q2_identify.py.

    python q2_grid_identify.py --selftest
    python q2_grid_identify.py --true F:0.35 R:0.1 --sessions 20 --hands 200
    python q2_grid_identify.py --true R:0.1 O:0 --compare-old 0.35     # the Week-4 example 4.10 case

--true gives seat 1 then seat 2 as TYPE:ALPHA (alphas need not be on the grid).
--compare-old A runs the old 16-pair identifier, which assumes alpha = A, on the same hands.
"""

import argparse
import itertools
import json
import pickle
import time

import numpy as np
import pyspiel

from cfr_path import find_cfr
from q2_identify import FAV, Identifier, perturb, play_hand

TYPES = ("R", "C", "F")
REGIMES = ("actions", "showdown", "critic")
CHECKPOINTS = (10, 25, 50, 100, 150, 200, 300)


class GridIdentifier:
    def __init__(self, game, base, grid=None, p_zero=0.25, seat=0, tremble=0.0, forget=1.0, switch=0.0):
        """seat: the seat doing the identifying (its own card is known; the other two are
        modelled). tremble: probability eps that a modelled opponent plays uniformly at
        random, mixed into every likelihood, so one near-impossible action cannot rule a
        hypothesis out forever. 0 reproduces the original identifier exactly.

        For opponents whose style changes, two ways to let old evidence fade (use one):
          forget  exponential forgetting: each hand, accumulated log-evidence is multiplied by
                  forget (< 1) before the new hand's is added; memory ~ 1 / (1 - forget) hands
          switch  fixed-share (Herbster & Warmuth 1998): each hand, each opponent independently
                  re-draws its type from the prior with probability switch; exact Bayes for that
                  model of change, and no hypothesis falls below ~ switch * prior
        Defaults (1.0, 0.0) reproduce the stationary identifier exactly."""
        if forget < 1.0 and switch > 0.0:
            raise ValueError("use forget or switch, not both")
        self.game, self.base = game, base
        self.seat, self.tremble, self.forget, self.switch = seat, tremble, forget, switch
        self.opps = [o for o in range(3) if o != seat]
        self.grid = np.round(np.arange(0.05, 0.6001, 0.05), 4) if grid is None else np.asarray(grid, float)
        # Hypotheses for ONE opponent: index 0 = equilibrium, then (type, alpha) by type.
        self.hyp = [("O", 0.0)] + [(t, float(a)) for t in TYPES for a in self.grid]
        self.H = len(self.hyp)
        self.alphas = np.array([a for _, a in self.hyp])
        self.favs = np.array([-1] + [FAV[t] for t in TYPES for _ in self.grid])   # -1: no tilt
        self.pi = np.array([p_zero] + [(1 - p_zero) / (self.H - 1)] * (self.H - 1))   # one opponent's prior
        lp = np.log(self.pi)
        self.logprior = (lp[:, None] + lp[None, :]).ravel()                     # independent opponents
        self.loglik = {r: np.zeros(self.H * self.H) for r in REGIMES}           # stationary / forgetting
        self.logpost = {r: self.logprior.copy() for r in REGIMES}               # fixed-share state
        self.deck = [a for a, _ in game.new_initial_state().chance_outcomes()]

    def seat_factor(self, decisions):
        """Likelihood of one opponent's decisions in a hand, under each of its H hypotheses."""
        L = np.ones(self.H)
        for d, a in decisions:                       # d = equilibrium probabilities at the infoset
            s, tot = d.get(a, 0.0), sum(d.values())
            fav_legal = np.isin(self.favs, list(d))  # tilt applies only where favoured action is legal
            num = (1 - self.alphas) * s + self.alphas * (self.favs == a)
            den = (1 - self.alphas) * tot + self.alphas
            f = np.where(fav_legal, num / den, s)
            if self.tremble:
                f = (1 - self.tremble) * f + self.tremble / len(d)
            L *= f
        return L

    def update(self, history, true_cards, showdown_seats):
        self.absorb(self.evidence(history), true_cards, showdown_seats)

    def evidence(self, history):
        """Each possible deal to the two opponents, with each opponent's likelihood under every
        hypothesis. Identifiers with the same grid and tremble can share this (absorb)."""
        g = self.game
        o1, o2 = self.opps
        c0, board, st = history[self.seat], None, g.new_initial_state()     # our own card
        for i, a in enumerate(history):              # find the board card, if dealt
            if i >= 3 and st.is_chance_node():
                board = a
            st.apply_action(a)
        free = [c for c in self.deck if c != c0 and c != board]
        combos = []
        for c1, c2 in itertools.permutations(free, 2):   # every possible deal to the two opponents
            h = list(history); h[o1], h[o2] = c1, c2     # history[p] is seat p's private card
            st = g.new_initial_state(); dec = {o1: [], o2: []}
            for a in h:
                if not st.is_chance_node():
                    p = st.current_player()
                    if p in dec:
                        dec[p].append((self.base[st.information_state_string(p)], a))
                st.apply_action(a)
            combos.append((c1, c2, self.seat_factor(dec[o1]), self.seat_factor(dec[o2])))
        return combos

    def absorb(self, combos, true_cards, showdown_seats):
        o1, o2 = self.opps
        for r in REGIMES:
            like = np.zeros(self.H * self.H)
            for c1, c2, L1, L2 in combos:
                if r == "critic" and (c1, c2) != (true_cards[o1], true_cards[o2]):
                    continue
                if r == "showdown" and ((o1 in showdown_seats and c1 != true_cards[o1]) or
                                        (o2 in showdown_seats and c2 != true_cards[o2])):
                    continue
                like += np.outer(L1, L2).ravel()     # joint over the deal: sum of products
            with np.errstate(divide="ignore"):
                ll = np.log(like)
            if self.switch > 0:
                P = self.share(self._norm(self.logpost[r]))
                with np.errstate(divide="ignore"):
                    self.logpost[r] = np.log(P.ravel()) + ll
            elif self.forget < 1.0:
                self.loglik[r] = self.forget * self.loglik[r] + ll
            else:
                self.loglik[r] += ll

    @staticmethod
    def _norm(lp):
        return np.exp(lp - np.logaddexp.reduce(lp))

    def share(self, P):
        """Fixed-share step: each opponent independently re-draws from its prior w.p. switch.
        P is the joint posterior (flat or H x H); returns the H x H mixed posterior."""
        h, pi = self.switch, self.pi
        P = P.reshape(self.H, self.H)
        m1, m2 = P.sum(1), P.sum(0)                  # each opponent's marginal
        return ((1 - h) ** 2 * P + h * (1 - h) * (np.outer(m1, pi) + np.outer(pi, m2))
                + h * h * np.outer(pi, pi))

    def posterior(self, r="actions"):
        lp = self.logpost[r] if self.switch > 0 else self.logprior + self.loglik[r]
        return np.exp(lp - np.logaddexp.reduce(lp)).reshape(self.H, self.H)

    def summary(self, r="actions"):
        P = self.posterior(r)
        out = {}
        for k, M in ((self.opps[0], P.sum(1)), (self.opps[1], P.sum(0))):   # marginal per opponent seat
            types = {"O": M[0], **{t: M[1 + i * len(self.grid): 1 + (i + 1) * len(self.grid)].sum()
                                   for i, t in enumerate(TYPES)}}
            order = np.argsort(self.alphas); cdf = np.cumsum(M[order])
            q = lambda x: float(self.alphas[order][min(np.searchsorted(cdf, x), len(cdf) - 1)])
            out[k] = dict(types=types, alpha_mean=float(M @ self.alphas), alpha_90=(q(0.05), q(0.95)))
        return P, out

    def p_types(self, P, t1, t2):
        """Posterior probability that the opponents are of types t1 and t2 (any alpha)."""
        m = lambda t: np.array([h[0] == t for h in self.hyp])
        return float(P[np.ix_(m(t1), m(t2))].sum())


def parse_true(s):
    t, a = s.split(":"); a = float(a)
    return ("O", 0.0) if t == "O" or a == 0 else (t, a)


def selftest(game, base):
    gi = GridIdentifier(game, base)
    rng = np.random.default_rng(0)
    keys = list(base); worst = 0.0
    for t in TYPES:                                  # 1. formula == perturb() everywhere tested
        for a in (0.05, 0.35, 0.6):
            tab = perturb(base, t, a); j = gi.hyp.index((t, a))
            for key in rng.choice(keys, 300, replace=False):
                for act in base[key]:
                    worst = max(worst, abs(gi.seat_factor([(base[key], act)])[j] - tab[key][act]))
    print(f"1. tilt formula matches perturb() (max diff {worst:.1e}) ...... {'PASS' if worst < 1e-12 else 'FAIL'}")
    # 2. on the hypotheses both identifiers share (alpha 0 or 0.35), log-likelihood
    #    differences must agree with the old 16-pair identifier
    old = Identifier(game, {t: perturb(base, t, 0.35) for t in ("O", "R", "C", "F")})
    tabs = [base, perturb(base, "F", 0.35), perturb(base, "R", 0.35)]
    for _ in range(8):
        hist, cards, sd, _ = play_hand(game, tabs, rng)
        gi.update(hist, cards, sd); old.update(hist, cards, sd)
    idx = lambda t: 0 if t == "O" else gi.hyp.index((t, 0.35))
    worst = 0.0
    for r in REGIMES:
        g = np.array([gi.loglik[r].reshape(gi.H, gi.H)[idx(a), idx(b)] for a, b in itertools.product("ORCF", repeat=2)])
        o = old.logpost[r]
        ok = np.isfinite(g) & np.isfinite(o)
        worst = max(worst, np.abs((g - g[ok][0]) - (o - o[ok][0]))[ok].max())
    print(f"2. agrees with the old identifier on shared hypotheses (max diff {worst:.1e}) {'PASS' if worst < 1e-9 else 'FAIL'}")


def run(game, base, truth, sessions, hands, seed, regime, compare_old):
    (t1, a1), (t2, a2) = truth
    pol = [base, perturb(base, t1, a1) if t1 != "O" else base, perturb(base, t2, a2) if t2 != "O" else base]
    rng = np.random.default_rng(seed)
    rows = []
    for sess in range(sessions):
        gi = GridIdentifier(game, base)
        old = Identifier(game, {t: perturb(base, t, compare_old) for t in ("O", "R", "C", "F")}) if compare_old else None
        first90 = first90_old = None; trace = {}
        for n in range(1, hands + 1):
            hist, cards, sd, _ = play_hand(game, pol, rng)
            gi.update(hist, cards, sd)
            P, s = gi.summary(regime)
            pt = gi.p_types(P, t1, t2)
            if first90 is None and pt >= 0.9:
                first90 = n
            if old is not None:
                old.update(hist, cards, sd)
                po = old.posterior(regime)[["O", "R", "C", "F"].index(t1) * 4 + ["O", "R", "C", "F"].index(t2)]
                if first90_old is None and po >= 0.9:
                    first90_old = n
            if n in CHECKPOINTS or n == hands:
                trace[n] = dict(p_types=pt, a1=s[1]["alpha_mean"], a2=s[2]["alpha_mean"],
                                ci1=s[1]["alpha_90"], ci2=s[2]["alpha_90"],
                                p_old=(float(po) if old is not None else None))
        rows.append(dict(first90=first90, first90_old=first90_old, trace=trace))
    return rows


def report(truth, rows, hands, compare_old):
    (t1, a1), (t2, a2) = truth
    f = [r["first90"] for r in rows]
    med = lambda v: (f"{np.median([x if x is not None else hands + 1 for x in v]):.0f}"
                     + (f" ({sum(x is None for x in v)} of {len(v)} never)" if any(x is None for x in v) else ""))
    print(f"\ntrue opponents: seat 1 = {t1} at {a1}, seat 2 = {t2} at {a2}   ({len(rows)} sessions)")
    print(f"  hands until P(true TYPES) >= 0.9:  grid {med(f)}"
          + (f"   old (assumes alpha {compare_old}) {med([r['first90_old'] for r in rows])}" if compare_old else ""))
    print(f"  {'hands':>6} {'P(types)':>9} {'alpha1 est':>11} {'90% int.':>14} {'alpha2 est':>11} {'90% int.':>14}"
          + (f" {'P(old)':>8}" if compare_old else ""))
    for n in sorted(rows[0]["trace"]):
        T = [r["trace"][n] for r in rows]
        cov1 = np.mean([c["ci1"][0] <= a1 <= c["ci1"][1] for c in T]); cov2 = np.mean([c["ci2"][0] <= a2 <= c["ci2"][1] for c in T])
        print(f"  {n:>6} {np.median([c['p_types'] for c in T]):>9.3f} {np.median([c['a1'] for c in T]):>11.3f}"
              f" {'covers ' + format(cov1, '.0%'):>14} {np.median([c['a2'] for c in T]):>11.3f} {'covers ' + format(cov2, '.0%'):>14}"
              + (f" {np.median([c['p_old'] for c in T]):>8.3f}" if compare_old else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", nargs=2, default=["F:0.35", "R:0.1"])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--hands", type=int, default=200)
    ap.add_argument("--regime", choices=REGIMES, default="actions")
    ap.add_argument("--compare-old", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfr", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    game = pyspiel.load_game("leduc_poker", {"players": 3})
    base = pickle.load(open(find_cfr(a.cfr), "rb"))["table"]
    if a.selftest:
        return selftest(game, base)
    truth = [parse_true(x) for x in a.true]
    t0 = time.time()
    rows = run(game, base, truth, a.sessions, a.hands, a.seed, a.regime, a.compare_old)
    report(truth, rows, a.hands, a.compare_old)
    print(f"  [{time.time()-t0:.0f}s]")
    if a.out:
        json.dump(dict(truth=truth, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
