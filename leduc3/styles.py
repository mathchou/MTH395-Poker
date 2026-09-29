"""
Style opponents for 3-player Leduc Hold'em.

Taxonomy adapted from enfiyeci/amp3-poker-ai (style_library.py), which defines ten
PlayerType archetypes for 6-player NLHE. Their preflop logic uses the Chen formula
over two hole cards, which has no analogue in Leduc's single private card, so each
archetype is re-expressed here as a (looseness, aggression, trap_rate, bluff_rate)
parameter tuple chosen to reproduce the archetype's intended VPIP/PFR profile.

Note: in the upstream repo, RuleBasedStrategy._preflop_action indexes a thresholds
dict covering only 5 of the 10 PlayerType values, so RuleBased_{MANIAC, ROCK,
CALLING_STATION, TAG, LAG} raise KeyError at runtime. All ten archetypes are
implemented here.

Feature set matches theirs: VPIP, PFR, AFq, WTSD.

open_spiel leduc_poker actions: 0 = fold, 1 = call/check, 2 = raise.
Cards 0..7, rank = card // 2, giving 4 ranks in the 3-player game.
"""

from enum import IntEnum

import numpy as np
import pyspiel

FOLD, CALL, RAISE = 0, 1, 2
NUM_PLAYERS = 3

# open_spiel scales the Leduc deck with player count: 2 suits x (num_players + 1)
# ranks. So 3-player Leduc has 8 cards across 4 ranks, NOT the 6-card / 3-rank
# deck of the familiar 2-player game. Verified against pyspiel chance_outcomes().
NUM_RANKS = NUM_PLAYERS + 1        # 4
DECK_SIZE = 2 * NUM_RANKS          # 8

FEATURE_NAMES = ["vpip", "pfr", "afq", "wtsd"]
STYLE_DIM = len(FEATURE_NAMES)


class PlayerType(IntEnum):
    """Ten archetypes, names matching enfiyeci/amp3-poker-ai."""
    CONSERVATIVE = 0
    REGULAR = 1
    AGGRESSIVE = 2
    BLUFFING = 3
    DECEPTIVE = 4
    MANIAC = 5
    ROCK = 6
    CALLING_STATION = 7
    TAG = 8
    LAG = 9


# looseness, aggression, trap_rate, bluff_rate
ARCHETYPE_PARAMS = {
    PlayerType.CONSERVATIVE:    (0.25, 0.30, 0.00, 0.00),
    PlayerType.REGULAR:         (0.50, 0.45, 0.05, 0.05),
    PlayerType.AGGRESSIVE:      (0.60, 0.80, 0.00, 0.10),
    PlayerType.BLUFFING:        (0.55, 0.65, 0.00, 0.35),
    PlayerType.DECEPTIVE:       (0.45, 0.40, 0.40, 0.10),
    PlayerType.MANIAC:          (0.95, 0.97, 0.00, 0.60),
    PlayerType.ROCK:            (0.10, 0.35, 0.00, 0.00),
    PlayerType.CALLING_STATION: (0.92, 0.05, 0.00, 0.00),
    PlayerType.TAG:             (0.30, 0.85, 0.00, 0.12),
    PlayerType.LAG:             (0.80, 0.85, 0.00, 0.25),
}

# Held out of training, used only in Experiment B.
HELDOUT_TYPES = [
    PlayerType.CALLING_STATION,
    PlayerType.MANIAC,
    PlayerType.DECEPTIVE,
]
TRAINING_TYPES = [t for t in PlayerType if t not in HELDOUT_TYPES]


def make_game():
    return pyspiel.load_game("leduc_poker", {"players": NUM_PLAYERS})


def hand_strength(state, player):
    """Strength in [0, 1]. A round-2 pair dominates every unpaired holding."""
    priv = state.private_card(player)
    pub = state.public_card()
    rank = priv // 2
    top = NUM_RANKS - 1
    if pub is None or pub < 0:
        return rank / top
    if rank == pub // 2:
        return 1.0
    return 0.35 * (rank / top) + (0.15 if rank > pub // 2 else 0.0)


class ArchetypePolicy(pyspiel.Policy):
    """
    Rule-based opponent. Softmax over action scores keeps the policy mixed, which
    matters because a pure policy makes best-response values degenerate and the
    exploitability measurement uninformative.
    """

    def __init__(self, player_type, softness=0.12):
        super().__init__()
        self.player_type = player_type
        self.name = player_type.name
        (self.looseness, self.aggression,
         self.trap_rate, self.bluff_rate) = ARCHETYPE_PARAMS[player_type]
        self.softness = softness

    def action_probabilities(self, state, player_id=None):
        if player_id is None:
            player_id = state.current_player()
        legal = state.legal_actions(player_id)
        s = hand_strength(state, player_id)

        # Deceptive archetype under-represents strength with premium holdings.
        eff = s * (1.0 - self.trap_rate) if s >= 0.99 else s
        # Bluffing archetypes over-represent strength with the worst holdings.
        if s <= 0.05:
            eff = max(eff, self.bluff_rate)

        fold_thresh = 0.55 * (1.0 - self.looseness)
        raise_thresh = 1.0 - 0.85 * self.aggression

        score = {}
        if FOLD in legal:
            score[FOLD] = (fold_thresh - eff) * 3.0
        if CALL in legal:
            score[CALL] = 1.0 - abs(eff - (fold_thresh + raise_thresh) / 2.0) * 2.0
        if RAISE in legal:
            score[RAISE] = (eff - raise_thresh) * 3.0

        keys = list(score)
        z = np.array([score[k] for k in keys]) / self.softness
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return {k: float(v) for k, v in zip(keys, p)}


def training_styles():
    return [ArchetypePolicy(t) for t in TRAINING_TYPES]


def heldout_styles():
    return [ArchetypePolicy(t) for t in HELDOUT_TYPES]


def all_styles():
    return [ArchetypePolicy(t) for t in PlayerType]


# --------------------------------------------------------------- style features

class StyleTracker:
    """
    Accumulates the four style statistics, mapped from Hold'em to Leduc:

      VPIP -> frequency of putting money in voluntarily in round 1
      PFR  -> raise frequency in round 1
      AFq  -> raise / (raise + call) in round 2, the postflop analogue
      WTSD -> frequency of reaching showdown given round-1 entry

    Leduc subtlety: round 1 opens with no outstanding bet, so CALL is a free check
    and must NOT count toward VPIP. In Hold'em the blinds guarantee there is always
    something to call, which is why the standard definition can ignore this. Facing
    a bet is detectable as FOLD being legal. Without this correction every archetype
    reports VPIP near 1.0 and the feature carries no signal.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._n = {k: 0.0 for k in FEATURE_NAMES}
        self._d = {k: 0.0 for k in FEATURE_NAMES}
        self.hands = 0
        self._entered = False
        self._folded = False

    def _bump(self, k, num):
        self._n[k] += num
        self._d[k] += 1.0

    def observe_action(self, state, player, action):
        facing_bet = FOLD in state.legal_actions(player)
        if action == FOLD:
            self._folded = True
        if state.round() == 1:
            # Free checks are not voluntary money.
            voluntary = (action == RAISE) or (facing_bet and action == CALL)
            self._bump("vpip", 1.0 if voluntary else 0.0)
            self._bump("pfr", 1.0 if action == RAISE else 0.0)
            if voluntary:
                self._entered = True
        else:
            if action in (CALL, RAISE):
                self._bump("afq", 1.0 if action == RAISE else 0.0)

    def observe_hand_end(self, reached_round2):
        self.hands += 1
        if self._entered:
            showdown = reached_round2 and not self._folded
            self._bump("wtsd", 1.0 if showdown else 0.0)
        self._entered = False
        self._folded = False

    def features(self, prior=2.0):
        """Laplace-smoothed toward 0.5 so tiny samples degrade gracefully."""
        return np.array(
            [(self._n[k] + prior * 0.5) / (self._d[k] + prior) for k in FEATURE_NAMES],
            dtype=np.float32,
        )
