"""
Vectorised game-tree engine for 3-player Leduc.

Why this exists: OpenSpiel's CFR solver updates every player every iteration and
cannot hold some players fixed or add a hidden "mode" chance node. Restricted Nash
responses need both. This module flattens the 1.83M-node tree into numpy arrays once,
then does forward passes (reach probabilities), backward passes (values), best
responses, and CFR+ updates with array operations.

Strategies are flat arrays over (information set, action) pairs, one per seat.

    python tree_engine.py build     # enumerate the tree, cache to leduc3p_tree.npz
    python tree_engine.py verify    # check against OpenSpiel before trusting anything
"""

import pickle
import sys
import time

import numpy as np
import pyspiel

TREE_FILE = "leduc3p_tree.npz"
NP = 3


# ----------------------------------------------------------------- building

def build(path=TREE_FILE, game_name="leduc_poker", players=NP, trace=0):
    """
    Walk the whole game tree once and store it as flat numpy arrays.

    Every later computation (expected values, best responses, CFR) runs on these
    arrays instead of on OpenSpiel State objects, which is what makes it fast.

    Each node of the tree gets an integer id, in the order it is visited. For node n:
        parent[n]   id of the node one move earlier (-1 for the root)
        depth[n]    number of moves (including chance deals) from the root
        kind[n]     who made the move INTO node n: 0, 1, 2 = that seat; 3 = chance; -1 = root
        esa[n]      if a seat made that move: which (information set, action) pair it was
        eprob[n]    if chance made that move: its probability
        util[n]     payoffs to all seats, if n is terminal (zeros otherwise)

    path       where to save the .npz file
    game_name  any OpenSpiel game with a "players" parameter, e.g. "kuhn_poker" (tiny,
               good for the debugger) or "leduc_poker" (the real one, 1.83M nodes)
    trace      print the first `trace` nodes as they are visited (0 = silent)
    """
    # Load the game. Everything below works for any 3-player OpenSpiel game; Tree
    # assumes 3 seats (NP), so keep players = 3.
    game = pyspiel.load_game(game_name, {"players": players})

    # Per-node lists, appended to as each node is created. Index in the list = node id.
    parent, depth, kind, esa, eprob = [], [], [], [], []
    util_idx, util_val = [], []                   # ids of terminal nodes and their payoffs
    inode_player, inode_infoset = [], []          # per node: seat to act and its infoset id (-1 if none)

    # Information sets, per seat. An information set is everything a seat knows when it
    # must act (its own card, the board, the betting so far); OpenSpiel describes it with
    # a string. key2id[seat] maps that string to a small integer id, assigned in order of
    # first appearance.
    key2id = [dict() for _ in range(players)]
    actions_of = [[] for _ in range(players)]      # actions_of[seat][infoset id] = legal actions there

    def add(par, d, k, sa, pr):
        """Create node id len(parent) and fill in its per-node fields."""
        parent.append(par); depth.append(d); kind.append(k); esa.append(sa); eprob.append(pr)
        inode_player.append(-1); inode_infoset.append(-1)   # filled in later if a seat acts here
        return len(parent) - 1

    # The (information set, action) index of an edge is an offset into a flat array whose
    # layout is only known once ALL information sets have been seen. So while walking,
    # each edge remembers (infoset id of its parent, position of the action in that
    # infoset's legal-action list), and the flat index is computed after the walk.
    edge_iset = []                                  # per node: infoset id of the parent (-1 if chance/root)
    edge_apos = []                                  # per node: position of the action taken in the parent

    # Depth-first walk with an explicit stack (recursion would be too deep for Python at
    # 1.83M nodes). Each stack entry describes a node that has been REACHED but not yet
    # created:
    #   (state, parent id, depth, kind of the move into it, parent's infoset id,
    #    action position, chance probability, label of the move, for tracing)
    stack = [(game.new_initial_state(), -1, 0, -1, -1, -1, 1.0, "root")]
    if trace:
        print(f"{'node':>5} {'parent':>6} {'depth':>5}  {'moved by':<8} {'move':<10} {'prob':>6}  "
              f"{'now':<9} {'stack':>5}  infoset")
    while stack:
        st, par, d, k, iset, apos, pr, label = stack.pop()   # most recently pushed first: depth-first

        n = add(par, d, k, -1, pr)                  # this node's id; esa is filled in after the walk
        edge_iset.append(iset); edge_apos.append(apos)

        if trace and n < trace:
            who = {-1: "-", 3: "chance"}.get(k, f"seat {k}")
            now = "terminal" if st.is_terminal() else "chance" if st.is_chance_node() else f"seat {st.current_player()}"
            info = st.information_state_string(st.current_player()) if not (st.is_terminal() or st.is_chance_node()) else ""
            print(f"{n:>5} {par:>6} {d:>5}  {who:<8} {label:<10} {pr:>6.3f}  {now:<9} {len(stack):>5}  {info.replace(chr(10), ' ')[:60]}")

        if st.is_terminal():
            # A leaf: record the payoff to every seat. Nothing to push.
            util_idx.append(n); util_val.append(st.returns())
            continue

        if st.is_chance_node():
            # A deal: one child per possible card, with its probability. The move into
            # each child is a chance move (kind 3), so it has no information set.
            for a, p in st.chance_outcomes():
                stack.append((st.child(a), n, d + 1, 3, -1, -1, p, _move_label(st, a)))
            continue

        # A decision node for seat `pl`.
        pl = st.current_player()
        key = st.information_state_string(pl)       # what pl knows here
        legal = st.legal_actions()                  # e.g. [fold, call, raise] or a subset
        if key not in key2id[pl]:
            # First time we see this information set: give it the next id and remember its
            # legal actions. Every other node in the same information set has the same
            # legal actions, because the betting history (which decides them) is public.
            key2id[pl][key] = len(actions_of[pl])
            actions_of[pl].append(list(legal))
        iid = key2id[pl][key]
        inode_player[n] = pl; inode_infoset[n] = iid

        # One child per legal action. kind = pl, because seat pl makes the move into the
        # child; the edge remembers (infoset id, action position) for the flat index.
        for pos, a in enumerate(legal):
            stack.append((st.child(a), n, d + 1, pl, iid, pos, 1.0, _move_label(st, a)))

    # ---- after the walk: convert to arrays -------------------------------------------
    N = len(parent)

    # Flat (information set, action) layout, per seat. The actions of infoset 0 come
    # first, then those of infoset 1, and so on:
    #   off[i]   position where infoset i's actions start (off[-1] = total pairs)
    #   sai[j]   which infoset pair j belongs to
    #   saa[j]   which OpenSpiel action pair j is
    sa_offset, sa_infoset, sa_action = [], [], []
    for pl in range(players):
        sizes = [len(a) for a in actions_of[pl]]                  # legal actions per infoset
        off = np.zeros(len(actions_of[pl]) + 1, dtype=np.int64)
        off[1:] = np.cumsum(sizes)                                # running total = start positions
        sa_offset.append(off)
        sa_infoset.append(np.repeat(np.arange(len(actions_of[pl])), sizes))
        sa_action.append(np.array([a for acts in actions_of[pl] for a in acts]))

    # Now every seat's edge can get its flat index: start of its infoset + action position.
    kind = np.array(kind, dtype=np.int8)
    esa = np.full(N, -1, dtype=np.int64)            # -1 for chance edges and the root
    ei = np.array(edge_iset); ea = np.array(edge_apos)
    for pl in range(players):
        m = kind == pl                              # edges where seat pl moved
        esa[m] = sa_offset[pl][ei[m]] + ea[m]

    # Payoffs: zero rows for internal nodes, returns() for terminal ones.
    util = np.zeros((N, players))
    util[np.array(util_idx)] = np.array(util_val)

    # Information-set strings in id order, so a strategy table keyed by these strings can
    # be converted to and from the flat arrays (Tree.from_table).
    keys = [np.array(sorted(key2id[pl], key=key2id[pl].get), dtype=object) for pl in range(players)]

    np.savez(path, parent=np.array(parent), depth=np.array(depth), kind=kind, esa=esa,
             eprob=np.array(eprob), util=util,
             inode_player=np.array(inode_player), inode_infoset=np.array(inode_infoset),
             **{f"off{p}": sa_offset[p] for p in range(players)},
             **{f"sai{p}": sa_infoset[p] for p in range(players)},
             **{f"saa{p}": sa_action[p] for p in range(players)},
             **{f"keys{p}": keys[p] for p in range(players)})
    print(f"{N:,} nodes, infosets per seat {[len(a) for a in actions_of]}, "
          f"SA pairs {[len(x) for x in sa_action]}")


def _move_label(st, a):
    """Human-readable name of action a at state st (used only for tracing)."""
    try:
        return st.action_to_string(st.current_player(), a)
    except Exception:
        return str(a)


# ------------------------------------------------------------------- engine

class Tree:
    def __init__(self, path=TREE_FILE):
        z = np.load(path, allow_pickle=True)
        self.parent, self.depth, self.kind = z["parent"], z["depth"], z["kind"]
        self.esa, self.eprob, self.util = z["esa"], z["eprob"], z["util"]
        self.N = len(self.parent)
        self.off = [z[f"off{p}"] for p in range(NP)]
        self.sai = [z[f"sai{p}"] for p in range(NP)]
        self.saa = [z[f"saa{p}"] for p in range(NP)]
        self.keys = [list(z[f"keys{p}"]) for p in range(NP)]
        self.nI = [len(k) for k in self.keys]
        self.nSA = [len(a) for a in self.saa]
        self.nact = [np.diff(o) for o in self.off]
        self.maxd = int(self.depth.max())
        self.by_depth = [np.where(self.depth == d)[0] for d in range(self.maxd + 1)]
        self.kmask = [np.where(self.kind == p)[0] for p in range(NP)]
        self.cmask = np.where(self.kind == 3)[0]

    # ---- strategies ----
    def from_table(self, table, player):
        """Flat SA array for `player` from a {infoset key: {action: prob}} dict."""
        out = np.zeros(self.nSA[player])
        for i, k in enumerate(self.keys[player]):
            d = table[k]
            for j in range(self.off[player][i], self.off[player][i + 1]):
                out[j] = d.get(int(self.saa[player][j]), 0.0)
        return out

    def uniform(self, player):
        return 1.0 / self.nact[player][self.sai[player]]

    def pure(self, player, chosen_sa):
        s = np.zeros(self.nSA[player]); s[chosen_sa] = 1.0
        return s

    # ---- passes ----
    def edge_probs(self, sig, skip=None):
        p = np.ones(self.N)
        p[self.cmask] = self.eprob[self.cmask]
        for pl in range(NP):
            if pl != skip:
                idx = self.kmask[pl]
                p[idx] = sig[pl][self.esa[idx]]
        return p

    def reach(self, pe):
        r = np.zeros(self.N); r[0] = 1.0
        for d in range(1, self.maxd + 1):
            nd = self.by_depth[d]
            r[nd] = r[self.parent[nd]] * pe[nd]
        return r

    def values(self, pe, u):
        v = u.copy()
        for d in range(self.maxd, 0, -1):
            nd = self.by_depth[d]
            v += np.bincount(self.parent[nd], weights=pe[nd] * v[nd], minlength=self.N)
        return v

    def ev(self, sig):
        pe = self.edge_probs(sig)
        return np.array([self.values(pe, self.util[:, i])[0] for i in range(NP)])

    def cfv(self, sig, player, u):
        """Counterfactual value per SA for `player`, under utility vector u."""
        pe = self.edge_probs(sig)
        v = self.values(pe, u)
        rmi = self.reach(self.edge_probs(sig, skip=player))
        e = self.kmask[player]
        return np.bincount(self.esa[e], weights=rmi[self.parent[e]] * v[e],
                           minlength=self.nSA[player])

    def respond(self, sig, player, score_u, track_u):
        """
        Best response for `player` by backward induction over information sets,
        choosing actions that MAXIMISE score_u. Returns the chosen pure strategy and
        the root value of track_u under it. Pass score_u = -u0 for an adversary who
        minimises seat 0's payoff; score_u = u_player for a self-interested one.
        """
        pe = self.edge_probs(sig, skip=player)
        rmi = self.reach(pe)
        vs, vt = score_u.copy(), track_u.copy()
        chosen = np.zeros(self.nSA[player], dtype=bool)
        off, sai = self.off[player], self.sai[player]
        for d in range(self.maxd, 0, -1):
            nd = self.by_depth[d]
            mine = nd[self.kind[nd] == player]
            if len(mine):
                sc = np.bincount(self.esa[mine], weights=rmi[self.parent[mine]] * vs[mine],
                                 minlength=self.nSA[player])
                touched = np.zeros(self.nSA[player], dtype=bool); touched[self.esa[mine]] = True
                sc_m = np.where(touched, sc, -np.inf)
                best = np.maximum.reduceat(sc_m, off[:-1])
                is_best = touched & (sc_m >= best[sai] - 1e-12)
                idx = np.where(is_best, np.arange(self.nSA[player]), self.nSA[player])
                first = np.minimum.reduceat(idx, off[:-1])
                ok = first < self.nSA[player]
                chosen[first[ok]] = True
                pe[mine] = chosen[self.esa[mine]].astype(float)
            vs += np.bincount(self.parent[nd], weights=pe[nd] * vs[nd], minlength=self.N)
            vt += np.bincount(self.parent[nd], weights=pe[nd] * vt[nd], minlength=self.N)
        return chosen.astype(float), vt[0]

    # ---- regret matching ----
    def rm(self, R, player):
        pos = np.maximum(R, 0)
        s = np.bincount(self.sai[player], weights=pos, minlength=self.nI[player])[self.sai[player]]
        return np.where(s > 0, pos / np.where(s > 0, s, 1), 1.0 / self.nact[player][self.sai[player]])

    def normalize(self, W, player):
        s = np.bincount(self.sai[player], weights=W, minlength=self.nI[player])[self.sai[player]]
        return np.where(s > 0, W / np.where(s > 0, s, 1), 1.0 / self.nact[player][self.sai[player]])


# ------------------------------------------------------------------- verify

def verify():
    from q1_alpha_sweep import merge
    T = Tree()
    from cfr_path import find_cfr
    base = pickle.load(open(find_cfr(), "rb"))["table"]
    sig = [T.from_table(base, p) for p in range(NP)]
    t0 = time.time()
    ev = T.ev(sig)
    print(f"engine EV (all CFR): {np.round(ev, 6)}   [{time.time()-t0:.1f}s]")
    game = pyspiel.load_game("leduc_poker", {"players": NP})
    ref = pyspiel.expected_returns(game.new_initial_state(),
                                   pyspiel.TabularPolicy(merge({0: base, 1: base, 2: base})),
                                   -1, True)
    print(f"OpenSpiel EV:        {np.round(ref, 6)}")
    t0 = time.time()
    _, v1 = T.respond(sig, 1, T.util[:, 1], T.util[:, 1])
    br = pyspiel.TabularBestResponse(game, 1, merge({0: base, 1: base, 2: base}))
    print(f"seat-1 BR value engine {v1:+.6f}  OpenSpiel {br.value_from_state(game.new_initial_state()):+.6f}"
          f"   [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    {"build": build, "verify": verify}[sys.argv[1]]()
