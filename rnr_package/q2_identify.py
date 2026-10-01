"""
Question 2: how many hands until we can classify the opponents?

We sit in seat 0 and play the CFR (optimal) policy. The two opponents are a
fixed pair drawn from {O, R, C, F} x {O, R, C, F} at a known alpha. We keep an
exact Bayesian posterior over the 16 pair hypotheses and update it after every
hand.

The likelihood of a hand under a pair hypothesis (t1, t2):

    P(observed actions | t1, t2)
        = sum over hole cards (c1, c2) consistent with what we saw
              P(c1, c2 | our card, board)
              * prod over seat-1 decisions  sigma_t1(a | I_1(c1, history))
              * prod over seat-2 decisions  sigma_t2(a | I_2(c2, history))

Our own action probabilities and the chance probabilities are identical under
every hypothesis, so they cancel. P(c1, c2 | ...) is uniform over distinct cards
not ours and not the board. This is exactly "compare what they did with what the
optimal policy would do given each possible hole card", weighted correctly.

Three information regimes, computed from the SAME simulated hands:

    actions   never see opponents' cards (not even at showdown)
    showdown  see the cards of opponents who reach showdown (real poker)
    critic    see both opponents' cards after every hand

    python q2_identify.py --alpha 0.35 --sessions 30 --hands 150
"""

import argparse
import itertools
import json
import pickle
import time

import numpy as np
import pyspiel
from cfr_path import find_cfr

FAV = {"R": 2, "C": 1, "F": 0}
TYPES = ["O", "R", "C", "F"]
PAIRS = [a + b for a, b in itertools.product(TYPES, repeat=2)]
REGIMES = ["actions", "showdown", "critic"]
CHECKPOINTS = [5, 10, 25, 50, 100, 150, 200, 300]


def perturb(base, t, alpha):
    if t == "O":
        return base
    fav = FAV[t]
    out = {}
    for k, d in base.items():
        if fav not in d:
            out[k] = d
            continue
        n = {a: (1 - alpha) * p for a, p in d.items()}
        n[fav] += alpha
        s = sum(n.values())
        out[k] = {a: p / s for a, p in n.items()}
    return out


class Identifier:
    def __init__(self, game, tables):
        self.game = game
        self.tables = tables                      # type -> table
        self.deck = [a for a, _ in game.new_initial_state().chance_outcomes()]
        self.logpost = {r: np.full(16, -np.log(16)) for r in REGIMES}

    def update(self, history, true_cards, showdown_seats):
        """history: full action list of a finished hand (first 3 = deals to seats 0,1,2)."""
        g = self.game
        c0 = history[0]
        # board card is the 4th chance action if the hand reached round 2
        s = g.new_initial_state()
        board, idx = None, 0
        for i, a in enumerate(history):
            if i >= 3 and s.is_chance_node():
                board = a
            s.apply_action(a)
        free = [c for c in self.deck if c != c0 and c != board]

        # per combo: likelihood of seat-1 and seat-2 actions under each type
        combos = []
        for c1, c2 in itertools.permutations(free, 2):
            h = list(history)
            h[1], h[2] = c1, c2
            st = g.new_initial_state()
            L1 = np.ones(4)
            L2 = np.ones(4)
            for a in h:
                if not st.is_chance_node():
                    p = st.current_player()
                    if p in (1, 2):
                        key = st.information_state_string(p)
                        L = L1 if p == 1 else L2
                        for j, t in enumerate(TYPES):
                            L[j] *= self.tables[t][key].get(a, 0.0)
                st.apply_action(a)
            combos.append((c1, c2, L1, L2))

        for r in REGIMES:
            like = np.zeros(16)
            for c1, c2, L1, L2 in combos:
                if r == "critic" and (c1, c2) != (true_cards[1], true_cards[2]):
                    continue
                if r == "showdown":
                    if 1 in showdown_seats and c1 != true_cards[1]:
                        continue
                    if 2 in showdown_seats and c2 != true_cards[2]:
                        continue
                like += np.outer(L1, L2).ravel()
            with np.errstate(divide="ignore"):
                lp = self.logpost[r] + np.log(like)
            lp -= np.logaddexp.reduce(lp)
            self.logpost[r] = lp

    def posterior(self, r):
        return np.exp(self.logpost[r])


def play_hand(game, policies, rng):
    s = game.new_initial_state()
    folded = set()
    while not s.is_terminal():
        if s.is_chance_node():
            a_, p_ = zip(*s.chance_outcomes())
            s.apply_action(int(rng.choice(a_, p=p_)))
            continue
        p = s.current_player()
        d = policies[p][s.information_state_string(p)]
        ks = list(d)
        a = int(rng.choice(ks, p=[d[k] for k in ks]))
        if a == 0:
            folded.add(p)
        s.apply_action(a)
    hist = s.history()
    alive = {p for p in range(3) if p not in folded}
    showdown = alive if len(alive) >= 2 else set()
    return hist, {p: hist[p] for p in range(3)}, showdown, s.returns()


def run(alpha, true_pairs, sessions, hands, seed, base):
    game = pyspiel.load_game("leduc_poker", {"players": 3})
    tables = {t: perturb(base, t, alpha) for t in TYPES}
    rng = np.random.default_rng(seed)
    out = {}
    for tp in true_pairs:
        ti = PAIRS.index(tp)
        stats = {r: {"p_true": {n: [] for n in CHECKPOINTS if n <= hands},
                     "map_ok": {n: [] for n in CHECKPOINTS if n <= hands},
                     "first90": []} for r in REGIMES}
        for _ in range(sessions):
            idf = Identifier(game, tables)
            pols = [base, tables[tp[0]], tables[tp[1]]]
            first90 = {r: None for r in REGIMES}
            for n in range(1, hands + 1):
                hist, cards, sd, _ = play_hand(game, pols, rng)
                idf.update(hist, cards, sd)
                for r in REGIMES:
                    post = idf.posterior(r)
                    if first90[r] is None and post[ti] >= 0.9:
                        first90[r] = n
                    if n in stats[r]["p_true"]:
                        stats[r]["p_true"][n].append(float(post[ti]))
                        stats[r]["map_ok"][n].append(int(np.argmax(post) == ti))
            for r in REGIMES:
                stats[r]["first90"].append(first90[r])
        out[tp] = stats
    return out


def summarize(res, hands):
    cps = [n for n in CHECKPOINTS if n <= hands]
    for tp, stats in res.items():
        print(f"\n true pair {tp}")
        print(f"  {'regime':<9}" + "".join(f"{'n=' + str(n):>8}" for n in cps)
              + f"{'median n to 90%':>17}")
        for r in REGIMES:
            acc = [np.mean(stats[r]["map_ok"][n]) for n in cps]
            f = [x for x in stats[r]["first90"] if x is not None]
            reached = len(f) / len(stats[r]["first90"])
            med = f"{int(np.median(f))}" if reached >= 0.5 else f">{hands}"
            print(f"  {r:<9}" + "".join(f"{a:>8.2f}" for a in acc)
                  + f"{med:>12} ({reached:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfr", default=None)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--pairs", nargs="+", default=["OO", "RO", "CO", "FO"])
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--hands", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = pickle.load(open(find_cfr(args.cfr), "rb"))["table"]
    t0 = time.time()
    res = run(args.alpha, args.pairs, args.sessions, args.hands, args.seed, base)
    print(f"alpha {args.alpha}, {args.sessions} sessions x {args.hands} hands "
          f"per pair  [{time.time()-t0:.0f}s]")
    print("table: fraction of sessions where the MAP pair is the true pair")
    summarize(res, args.hands)
    if args.out:
        json.dump(res, open(args.out, "w"))


if __name__ == "__main__":
    main()
