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

def build(path=TREE_FILE):
    game = pyspiel.load_game("leduc_poker", {"players": NP})
    parent, depth, kind, esa, eprob = [], [], [], [], []
    util_idx, util_val = [], []
    inode_player, inode_infoset = [], []          # per node: acting seat / infoset id
    key2id = [dict() for _ in range(NP)]
    actions_of = [[] for _ in range(NP)]           # infoset id -> list of legal actions

    def add(par, d, k, sa, pr):
        parent.append(par); depth.append(d); kind.append(k); esa.append(sa); eprob.append(pr)
        inode_player.append(-1); inode_infoset.append(-1)
        return len(parent) - 1

    # sa index needs offsets that are only known once all infosets exist, so edges store
    # (infoset id, action position) first and are converted at the end.
    edge_iset = []                                  # per node: infoset id of parent (or -1)
    edge_apos = []                                  # per node: action position in parent

    stack = [(game.new_initial_state(), -1, 0, -1, -1, -1, 1.0)]
    while stack:
        st, par, d, k, iset, apos, pr = stack.pop()
        n = add(par, d, k, -1, pr)
        edge_iset.append(iset); edge_apos.append(apos)
        if st.is_terminal():
            util_idx.append(n); util_val.append(st.returns())
            continue
        if st.is_chance_node():
            for a, p in st.chance_outcomes():
                stack.append((st.child(a), n, d + 1, 3, -1, -1, p))
            continue
        pl = st.current_player()
        key = st.information_state_string(pl)
        legal = st.legal_actions()
        if key not in key2id[pl]:
            key2id[pl][key] = len(actions_of[pl])
            actions_of[pl].append(list(legal))
        iid = key2id[pl][key]
        inode_player[n] = pl; inode_infoset[n] = iid
        for pos, a in enumerate(legal):
            stack.append((st.child(a), n, d + 1, pl, iid, pos, 1.0))

    N = len(parent)
    sa_offset, sa_infoset, sa_action = [], [], []
    for pl in range(NP):
        off = np.zeros(len(actions_of[pl]) + 1, dtype=np.int64)
        off[1:] = np.cumsum([len(a) for a in actions_of[pl]])
        sa_offset.append(off)
        sa_infoset.append(np.repeat(np.arange(len(actions_of[pl])),
                                    [len(a) for a in actions_of[pl]]))
        sa_action.append(np.array([a for acts in actions_of[pl] for a in acts]))

    kind = np.array(kind, dtype=np.int8)
    esa = np.full(N, -1, dtype=np.int64)
    ei = np.array(edge_iset); ea = np.array(edge_apos)
    for pl in range(NP):
        m = kind == pl
        esa[m] = sa_offset[pl][ei[m]] + ea[m]
    util = np.zeros((N, NP))
    util[np.array(util_idx)] = np.array(util_val)
    keys = [np.array(sorted(key2id[pl], key=key2id[pl].get), dtype=object) for pl in range(NP)]

    np.savez(path, parent=np.array(parent), depth=np.array(depth), kind=kind, esa=esa,
             eprob=np.array(eprob), util=util,
             inode_player=np.array(inode_player), inode_infoset=np.array(inode_infoset),
             **{f"off{p}": sa_offset[p] for p in range(NP)},
             **{f"sai{p}": sa_infoset[p] for p in range(NP)},
             **{f"saa{p}": sa_action[p] for p in range(NP)},
             **{f"keys{p}": keys[p] for p in range(NP)})
    print(f"{N:,} nodes, infosets per seat {[len(a) for a in actions_of]}, "
          f"SA pairs {[len(x) for x in sa_action]}")


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
    base = pickle.load(open("cfr_3p_1000.pkl", "rb"))["table"]
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
