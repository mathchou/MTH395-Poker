"""
AMP3 training loop for 3-player Leduc.

Asymmetric actor-critic, following Shi et al. (2022): the Actor sees only the
information set, the Critic sees all private cards. Returns are Monte Carlo over
complete hands, since a Leduc hand is at most about six decisions and there is no
intermediate reward, so bootstrapping buys nothing.

Two modes:
  --style none    unconditioned agent (the week-4 deliverable)
  --style oracle  conditioned on the opponents' true style features

Oracle conditioning is deliberate: it isolates whether the RL machinery works at
all from whether OSM can estimate style. If the oracle-conditioned agent does not
beat the unconditioned one, nothing downstream of it is worth building.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import encode
from styles import (make_game, training_styles, ArchetypePolicy, PlayerType,
                    ARCHETYPE_PARAMS, NUM_PLAYERS, FEATURE_NAMES)
from networks import AMP3Network, OSMNetwork, FULL_STYLE_DIM
from evaluate import expected_returns_exact
from tabularize import tabularize

STYLE_DIM = len(FEATURE_NAMES)
# Perfect-information critic input: every private card + board + flag + stacks + pot
PERFECT_DIM = NUM_PLAYERS * (encode.NUM_RANKS) + encode.PUBLIC_DIM


class PerfectCritic(nn.Module):
    """Privileged value function. Sees all hands; used only during training."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PERFECT_DIM + encode.POSITION_DIM, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_perfect(state, player):
    cards = [encode._rank_onehot(state.private_card(p)) for p in range(NUM_PLAYERS)]
    return np.concatenate(cards + [encode.encode_public(state)]).astype(np.float32)


POOL_MAXSEQ = 30   # must match build_osm.MAX_SEQ

_MEAS = None


def measured_style_vector(opponents):
    """
    Ground-truth MEASURED features of the seated opponents, from
    measured_features.npz. This is the honest oracle: it is exactly what a perfect
    OSM would output, so it upper-bounds what style estimation can deliver.
    Contrast true_style_vector(), which reads generative parameters instead.
    """
    global _MEAS
    if _MEAS is None:
        z = np.load("measured_features.npz", allow_pickle=True)
        _MEAS = {str(n): f for n, f in zip(z["names"], z["feats"])}
    return np.concatenate([_MEAS[o.player_type.name] for o in opponents]).astype(np.float32)


class OSMEstimator:
    """Wraps a trained OSM net. Estimates style from a session action history."""

    def __init__(self, path="osm.pt"):
        self.net = OSMNetwork()
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    def estimate(self, hist, opp_seats, state):
        """hist: list of (player, action) across the session so far."""
        if not hist:
            return np.full(FULL_STYLE_DIM, 0.5, dtype=np.float32)
        pub = torch.tensor(encode.encode_public(state)).unsqueeze(0)
        out = []
        for seat in opp_seats:
            ah = torch.tensor(
                encode.encode_action_history(hist, seat, max_seq=POOL_MAXSEQ)
            ).unsqueeze(0)
            with torch.no_grad():
                out.append(self.net(ah, pub)[0].numpy())
        return np.concatenate(out).astype(np.float32)


def true_style_vector(opponents):
    """Ground-truth style features for the two opponents, from ARCHETYPE_PARAMS."""
    v = []
    for op in opponents:
        loose, aggr, trap, bluff = ARCHETYPE_PARAMS[op.player_type]
        # crude but monotone stand-ins for vpip / pfr / afq / wtsd
        v.extend([loose, aggr * (1 - trap), aggr, 0.5 + 0.3 * loose])
    return np.array(v, dtype=np.float32)


class Agent:
    def __init__(self, use_style, lr=3e-4, ent_coef=0.01, seed=0, permuted=False,
                 osm=None, session=10, measured=False):
        torch.manual_seed(seed)
        self.use_style = use_style
        self.permuted = permuted
        self.osm = osm
        self.session = session
        self.measured = measured
        self.actor = AMP3Network()
        self.critic = PerfectCritic()
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.ent_coef = ent_coef

    def _tensors(self, state, player, seq, style):
        per = torch.tensor(encode.encode_personal(state, player)).unsqueeze(0)
        pub = torch.tensor(encode.encode_public(state)).unsqueeze(0)
        pos = torch.tensor(encode.encode_position(player)).unsqueeze(0)
        ah = torch.tensor(encode.encode_action_history(seq, player)).unsqueeze(0)
        st = torch.tensor(style if self.use_style
                          else np.zeros(FULL_STYLE_DIM, dtype=np.float32)).unsqueeze(0)
        return per, pub, pos, ah, st

    def policy_dist(self, state, player, seq, style):
        per, pub, pos, ah, st = self._tensors(state, player, seq, style)
        mask = torch.zeros(1, encode.NUM_ACTIONS, dtype=torch.bool)
        for a in state.legal_actions(player):
            mask[0, a] = True
        logits, _ = self.actor(per, pub, pos, ah, st, legal_mask=mask)
        return torch.distributions.Categorical(logits=logits[0]), (per, pub, pos, ah, st, mask)


def collect(agent, game, style_pool, rng, n_hands):
    """Roll out n_hands with the agent in a rotating seat. Returns a batch."""
    obs, acts, rets, perf = [], [], [], []
    session_hist, session_left, opps, seat = [], 0, None, 0
    for h in range(n_hands):
        if session_left == 0:
            seat = (h // max(agent.session, 1)) % NUM_PLAYERS
            opps = [style_pool[rng.integers(len(style_pool))] for _ in range(2)]
            session_hist, session_left = [], agent.session
        session_left -= 1
        if agent.permuted:
            # Experiment A control: same dimensionality and marginal distribution,
            # zero mutual information with the opponents actually seated.
            fake = [style_pool[rng.integers(len(style_pool))] for _ in range(2)]
            style = true_style_vector(fake)
        elif agent.measured:
            style = measured_style_vector(opps)
        elif agent.osm is not None:
            style = agent.osm.estimate(
                session_hist, [p for p in range(NUM_PLAYERS) if p != seat],
                game.new_initial_state().child(0))
        else:
            style = true_style_vector(opps)
        seats = {}
        oi = 0
        for p in range(NUM_PLAYERS):
            if p == seat:
                seats[p] = None
            else:
                seats[p] = opps[oi]; oi += 1

        s = game.new_initial_state()
        seq, mine = [], []
        while not s.is_terminal():
            if s.is_chance_node():
                a_, p_ = zip(*s.chance_outcomes())
                s.apply_action(int(rng.choice(a_, p=p_)))
                continue
            p = s.current_player()
            if p == seat:
                dist, tens = agent.policy_dist(s, p, seq, style)
                a = int(dist.sample())
                mine.append((tens, a, encode_perfect(s, p), encode.encode_position(p)))
            else:
                pr = seats[p].action_probabilities(s, p)
                ks = list(pr)
                a = int(rng.choice(ks, p=[pr[k] for k in ks]))
            seq.append((p, a))
            s.apply_action(a)
        session_hist.extend(seq)
        session_hist = session_hist[-POOL_MAXSEQ:]

        R = s.returns()[seat]
        for tens, a, pf, po in mine:
            obs.append(tens); acts.append(a); rets.append(R)
            perf.append(np.concatenate([pf, po]))
    return obs, acts, np.array(rets, dtype=np.float32), np.array(perf, dtype=np.float32)


def update(agent, obs, acts, rets, perf):
    per = torch.cat([o[0] for o in obs]); pub = torch.cat([o[1] for o in obs])
    pos = torch.cat([o[2] for o in obs]); ah = torch.cat([o[3] for o in obs])
    st = torch.cat([o[4] for o in obs]); mask = torch.cat([o[5] for o in obs])
    a = torch.tensor(acts); R = torch.tensor(rets)
    P = torch.tensor(perf)

    logits, _ = agent.actor(per, pub, pos, ah, st, legal_mask=mask)
    dist = torch.distributions.Categorical(logits=logits)
    V = agent.critic(P)
    adv = (R - V).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)

    pg = -(dist.log_prob(a) * adv).mean()
    vl = nn.functional.mse_loss(V, R)
    ent = dist.entropy().mean()
    loss = pg + 0.5 * vl - agent.ent_coef * ent

    agent.opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(agent.actor.parameters()) + list(agent.critic.parameters()), 1.0)
    agent.opt.step()
    return float(pg.detach()), float(vl.detach()), float(ent.detach())


class AgentPolicy:
    """Wraps a trained agent as an open_spiel-compatible policy at a fixed style."""

    def __init__(self, agent, style):
        self.agent = agent
        self.style = style

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        with torch.no_grad():
            dist, _ = self.agent.policy_dist(state, player_id, [], self.style)
            p = dist.probs
        return {a: float(p[a]) for a in state.legal_actions(player_id)}


def exact_eval(agent, game, opponent, style):
    pol = tabularize(game, AgentPolicy(agent, style), verbose=False)
    return expected_returns_exact(game, [pol, opponent, opponent])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", choices=["none", "oracle", "permuted", "osm", "measured"], default="none")
    ap.add_argument("--session", type=int, default=10)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--hands", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval-every", type=int, default=25)
    args = ap.parse_args()

    game = make_game()
    rng = np.random.default_rng(args.seed)
    pool = training_styles()
    osm = OSMEstimator() if args.style == "osm" else None
    agent = Agent(use_style=(args.style != "none"), lr=args.lr,
                  seed=args.seed, permuted=(args.style == "permuted"),
                  osm=osm, session=args.session,
                  measured=(args.style == "measured"))
    agent.ent_coef = args.ent
    import os
    if args.ckpt and os.path.exists(args.ckpt):
        sd = torch.load(args.ckpt)
        agent.actor.load_state_dict(sd["actor"]); agent.critic.load_state_dict(sd["critic"])
        agent.opt.load_state_dict(sd["opt"])
        print(f"resumed from {args.ckpt}", flush=True)

    regular = ArchetypePolicy(PlayerType.REGULAR)
    if args.style == "osm":
        hist = []
        for _ in range(5):
            st = game.new_initial_state()
            while not st.is_terminal():
                if st.is_chance_node():
                    a_, p_ = zip(*st.chance_outcomes())
                    st.apply_action(int(rng.choice(a_, p=p_))); continue
                pp = st.current_player()
                pr = regular.action_probabilities(st, pp); ks = list(pr)
                aa = int(rng.choice(ks, p=[pr[k] for k in ks]))
                hist.append((pp, aa)); st.apply_action(aa)
        eval_style = osm.estimate(hist[-POOL_MAXSEQ:], [1, 2],
                                  game.new_initial_state().child(0))
    elif args.style == "measured":
        eval_style = measured_style_vector([regular, regular])
    else:
        eval_style = true_style_vector([regular, regular])

    print(f"mode={args.style} updates={args.updates} hands/update={args.hands}", flush=True)
    t0 = time.time()
    for u in range(1, args.updates + 1):
        obs, acts, rets, perf = collect(agent, game, pool, rng, args.hands)
        pg, vl, ent = update(agent, obs, acts, rets, perf)
        if u % 10 == 0:
            print(f"  u{u:4d}  mean_return {rets.mean():+.3f}  pg {pg:+.4f} "
                  f"vloss {vl:.3f}  ent {ent:.3f}  [{time.time()-t0:.0f}s]", flush=True)
        if u % args.eval_every == 0 or u == args.updates:
            ev = exact_eval(agent, game, regular, eval_style)
            print(f"  u{u:4d}  EXACT EV vs 2x REGULAR: {ev:+.4f} chips/hand", flush=True)

    ev = exact_eval(agent, game, regular, eval_style)
    print(f"FINAL exact EV vs 2x REGULAR: {ev:+.4f}", flush=True)
    if args.ckpt:
        torch.save({"actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "opt": agent.opt.state_dict()}, args.ckpt)


if __name__ == "__main__":
    main()
