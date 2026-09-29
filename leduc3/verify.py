"""
Verification suite. Every check either passes or reports why not.

  V1  All ten archetypes execute without error (the upstream failure mode).
  V2  Archetypes are behaviourally differentiated in exact expected value.
  V3  Measured style features match each archetype's intended profile.
  V4  Network shapes are consistent and forward passes run.
  V5  OSM can actually learn to recover style from action history.
  V6  A network policy integrates with open_spiel and nash_conv runs on it.
"""

import sys
import time

import numpy as np
import torch

import encode
from styles import (make_game, all_styles, training_styles, heldout_styles,
                    ArchetypePolicy, PlayerType, StyleTracker, FEATURE_NAMES,
                    FOLD, CALL, RAISE)
from networks import OSMNetwork, AMP3Network, FULL_STYLE_DIM
from evaluate import expected_returns_exact

PASS, FAIL = "PASS", "FAIL"
results = []


def report(name, ok, detail=""):
    results.append((name, PASS if ok else FAIL, detail))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}", flush=True)


# ------------------------------------------------------------------ V1
def v1_all_archetypes_run():
    game = make_game()
    rng = np.random.default_rng(0)
    crashed = []
    for pol in all_styles():
        try:
            for _ in range(30):
                s = game.new_initial_state()
                while not s.is_terminal():
                    if s.is_chance_node():
                        acts, probs = zip(*s.chance_outcomes())
                        s.apply_action(rng.choice(acts, p=probs))
                    else:
                        pr = pol.action_probabilities(s, s.current_player())
                        acts = list(pr)
                        s.apply_action(rng.choice(acts, p=[pr[a] for a in acts]))
        except Exception as e:
            crashed.append((pol.name, type(e).__name__))
    report("V1 all 10 archetypes execute", not crashed,
           f"{len(crashed)} crashed" if crashed else "0 crashed (upstream: 5/37)")
    return crashed


# ------------------------------------------------------------------ V2
def v2_differentiation():
    game = make_game()
    ref = ArchetypePolicy(PlayerType.REGULAR)
    evs = {}
    for pol in all_styles():
        evs[pol.name] = expected_returns_exact(game, [pol, ref, ref])[0]
    vals = np.array(list(evs.values()))
    spread = vals.max() - vals.min()
    print("     exact EV vs two REGULAR opponents (chips/hand):")
    for k, v in sorted(evs.items(), key=lambda kv: -kv[1]):
        print(f"       {k:<18} {v:+.4f}")
    report("V2 archetypes differentiated", spread > 0.05,
           f"spread {spread:.4f}, std {vals.std():.4f}")
    return evs


# ------------------------------------------------------------------ V3
def play_hands(game, policies, n, rng, track_player=0):
    """Roll out n hands, returning the tracker and collected (history, public) data."""
    tracker = StyleTracker()
    samples = []
    for _ in range(n):
        s = game.new_initial_state()
        seq = []
        while not s.is_terminal():
            if s.is_chance_node():
                acts, probs = zip(*s.chance_outcomes())
                s.apply_action(rng.choice(acts, p=probs))
                continue
            p = s.current_player()
            pr = policies[p].action_probabilities(s, p)
            acts = list(pr)
            a = int(rng.choice(acts, p=[pr[x] for x in acts]))
            if p == track_player:
                tracker.observe_action(s, p, a)
                samples.append((encode.encode_action_history(seq, track_player),
                                encode.encode_public(s)))
            seq.append((p, a))
            s.apply_action(a)
        # showdown iff the tracked player was not folded out and round 2 was reached
        reached = s.round() == 2
        tracker.observe_hand_end(reached)
    return tracker, samples


def v3_feature_profiles():
    game = make_game()
    rng = np.random.default_rng(1)
    ref = ArchetypePolicy(PlayerType.REGULAR)
    print(f"     measured features over 3000 hands ({', '.join(FEATURE_NAMES)}):")
    feats = {}
    for pol in all_styles():
        tr, _ = play_hands(game, [pol, ref, ref], 3000, rng, track_player=0)
        f = tr.features()
        feats[pol.name] = f
        print(f"       {pol.name:<18} " + " ".join(f"{x:.3f}" for x in f))

    rock, station = feats["ROCK"], feats["CALLING_STATION"]
    maniac, conservative = feats["MANIAC"], feats["CONSERVATIVE"]
    tag = feats["TAG"]
    checks = {
        "ROCK vpip < CALLING_STATION vpip": rock[0] < station[0],
        "MANIAC pfr > CONSERVATIVE pfr": maniac[1] > conservative[1],
        "CALLING_STATION pfr < TAG pfr": station[1] < tag[1],
        "MANIAC vpip > ROCK vpip": maniac[0] > rock[0],
    }
    for k, v in checks.items():
        print(f"       {'ok ' if v else 'BAD'} {k}")
    report("V3 feature profiles match archetypes", all(checks.values()),
           f"{sum(checks.values())}/{len(checks)} ordering checks")
    return feats


# ------------------------------------------------------------------ V4
def v4_network_shapes():
    B = 7
    osm = OSMNetwork()
    amp3 = AMP3Network()
    ah = torch.randn(B, encode.MAX_SEQ, encode.ACTION_FEAT_DIM)
    pub = torch.randn(B, encode.PUBLIC_DIM)
    per = torch.randn(B, encode.PERSONAL_DIM)
    pos = torch.randn(B, encode.POSITION_DIM)
    sty = torch.rand(B, FULL_STYLE_DIM)
    mask = torch.ones(B, encode.NUM_ACTIONS, dtype=torch.bool)
    mask[:, 0] = False

    o = osm(ah, pub)
    logits, val = amp3(per, pub, pos, ah, sty, legal_mask=mask)
    probs = torch.softmax(logits, dim=-1)

    ok = (
        o.shape == (B, 4)
        and float(o.min()) >= 0.0 and float(o.max()) <= 1.0
        and logits.shape == (B, encode.NUM_ACTIONS)
        and val.shape == (B,)
        and torch.allclose(probs.sum(-1), torch.ones(B), atol=1e-5)
        and float(probs[:, 0].abs().max()) < 1e-8      # masked action has zero mass
    )
    n_osm = sum(p.numel() for p in osm.parameters())
    n_amp = sum(p.numel() for p in amp3.parameters())
    report("V4 network shapes and masking", ok,
           f"OSM {n_osm:,} params, AMP3 {n_amp:,} params, fusion dim {amp3.combined_dim}")


# ------------------------------------------------------------------ V5
def v5_osm_learns():
    """
    Train OSM to regress style features from pooled action history.

    A single Leduc hand yields only 2-4 actions from the target player, which is
    close to information-free; pooling five hands per input is the minimum that
    works. See osm_budget.py for the sweep. Sample count must be held constant
    across pooling levels or the comparison is confounded by training-set size.
    """
    from osm_budget import run as budget_run
    K, n, mse, base, red = budget_run(5, hps=3000)
    report("V5 OSM learns style from history", red > 40.0,
           f"pooling 5 hands: test MSE {mse:.5f} vs baseline {base:.5f} "
           f"({red:.1f}% reduction), n={n:,}")


# ------------------------------------------------------------------ V6
def v6_openspiel_integration():
    """Wrap AMP3 as an open_spiel Policy and confirm nash_conv accepts it."""
    import pyspiel
    from evaluate import MixedProfile, nash_conv_3p

    game = make_game()
    net = AMP3Network().eval()

    class NetPolicy(pyspiel.Policy):
        def __init__(self, net, style_vec):
            super().__init__()
            self.net = net
            self.style = torch.tensor(style_vec, dtype=torch.float32).unsqueeze(0)

        def action_probabilities(self, state, player_id=None):
            if player_id is None:
                player_id = state.current_player()
            legal = state.legal_actions(player_id)
            per = torch.tensor(encode.encode_personal(state, player_id)).unsqueeze(0)
            pub = torch.tensor(encode.encode_public(state)).unsqueeze(0)
            pos = torch.tensor(encode.encode_position(player_id)).unsqueeze(0)
            ah = torch.tensor(
                encode.encode_action_history([], player_id)).unsqueeze(0)
            mask = torch.zeros(1, encode.NUM_ACTIONS, dtype=torch.bool)
            for a in legal:
                mask[0, a] = True
            with torch.no_grad():
                logits, _ = self.net(per, pub, pos, ah, self.style, legal_mask=mask)
                p = torch.softmax(logits, -1)[0]
            return {a: float(p[a]) for a in legal}

    from tabularize import tabularize
    raw = NetPolicy(net, np.full(FULL_STYLE_DIM, 0.5, dtype=np.float32))
    pol = tabularize(game, raw)      # required: see tabularize.py
    ev = expected_returns_exact(game, [pol, pol, pol])
    zero_sum = abs(float(ev.sum())) < 1e-6

    t0 = time.time()
    nc = nash_conv_3p(game, MixedProfile([pol, pol, pol]))
    dt = time.time() - t0
    report("V6 open_spiel integration", zero_sum and nc > 0,
           f"returns sum {ev.sum():+.2e} (zero-sum ok), nash_conv {nc:.4f} in {dt:.0f}s")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"v1": v1_all_archetypes_run, "v2": v2_differentiation,
           "v3": v3_feature_profiles, "v4": v4_network_shapes,
           "v5": v5_osm_learns, "v6": v6_openspiel_integration}
    for k, f in fns.items():
        if which in ("all", k):
            f()
    print("\n" + "=" * 60)
    for n, s, d in results:
        print(f"{s:>4}  {n}")
