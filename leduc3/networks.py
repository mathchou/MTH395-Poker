"""
OSM and AMP3 networks for 3-player Leduc.

Ported from enfiyeci/amp3-poker-ai (osm_network.py, amp3_network.py). The
architectural pattern is preserved:

  - OSM: bidirectional LSTM over action history, concatenated with card context,
    regressing the four style features.
  - AMP3: late fusion. Separate encoders for personal / public / position / style,
    plus a bidirectional LSTM over action history. All branches concatenated into a
    single vector feeding policy and value heads.

Two deliberate departures from upstream:

  1. Sizes are reduced (hidden 64, LSTM 64, 2 layers) because Leduc's action
     sequences are at most about six steps and 25,800 information states do not
     support a 480-dim fusion vector without heavy overfitting.
  2. The Deep CFR and NFSP auxiliary heads in cfr_networks.py are omitted. They do
     not appear in the paper's description of AMP3 and would confound the
     exploitation-versus-exploitability measurement.

The style branch is kept structurally isolated so that Experiment A (permutation
control) can zero or shuffle it while holding parameter count exactly constant.
"""

import torch
import torch.nn as nn

from encode import (PERSONAL_DIM, PUBLIC_DIM, POSITION_DIM, ACTION_FEAT_DIM,
                    NUM_ACTIONS)
from styles import STYLE_DIM, NUM_PLAYERS

NUM_OPPONENTS = NUM_PLAYERS - 1
FULL_STYLE_DIM = NUM_OPPONENTS * STYLE_DIM   # 2 opponents x 4 features = 8


class OSMNetwork(nn.Module):
    """Predicts an opponent's four style features from observed action history."""

    def __init__(self, hidden=64, lstm_hidden=64, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=ACTION_FEAT_DIM,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.card_encoder = nn.Sequential(
            nn.Linear(PUBLIC_DIM, hidden), nn.LeakyReLU(0.1),
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2 + hidden, hidden), nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden, STYLE_DIM), nn.Sigmoid(),   # features live in [0,1]
        )

    def forward(self, action_history, public):
        _, (h, _) = self.lstm(action_history)
        seq = torch.cat([h[-2], h[-1]], dim=-1)           # both directions, top layer
        return self.head(torch.cat([seq, self.card_encoder(public)], dim=-1))


class AMP3Network(nn.Module):
    """Style-conditioned actor-critic. Late fusion, following upstream."""

    def __init__(self, hidden=64, lstm_hidden=64, lstm_layers=2, dropout=0.1):
        super().__init__()
        h2, h4 = hidden // 2, hidden // 4

        self.personal_encoder = nn.Sequential(
            nn.Linear(PERSONAL_DIM, hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden, h2), nn.LeakyReLU(0.1),
        )
        self.public_encoder = nn.Sequential(
            nn.Linear(PUBLIC_DIM, hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden, h2), nn.LeakyReLU(0.1),
        )
        self.position_encoder = nn.Sequential(
            nn.Linear(POSITION_DIM, h4), nn.LeakyReLU(0.1),
        )
        self.style_encoder = nn.Sequential(
            nn.Linear(FULL_STYLE_DIM, hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden, h2), nn.LeakyReLU(0.1),
        )
        self.action_lstm = nn.LSTM(
            input_size=ACTION_FEAT_DIM,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        combined = h2 * 3 + h4 + lstm_hidden * 2
        self.combined_dim = combined

        self.policy_head = nn.Sequential(
            nn.Linear(combined, hidden), nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(hidden, h2), nn.LeakyReLU(0.1),
            nn.Linear(h2, NUM_ACTIONS),
        )
        self.value_head = nn.Sequential(
            nn.Linear(combined, hidden), nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
        )

    def fuse(self, personal, public, position, action_history, style):
        _, (h, _) = self.action_lstm(action_history)
        seq = torch.cat([h[-2], h[-1]], dim=-1)
        return torch.cat([
            self.personal_encoder(personal),
            self.public_encoder(public),
            self.position_encoder(position),
            self.style_encoder(style),
            seq,
        ], dim=-1)

    def forward(self, personal, public, position, action_history, style,
                legal_mask=None):
        z = self.fuse(personal, public, position, action_history, style)
        logits = self.policy_head(z)
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float("-inf"))
        return logits, self.value_head(z).squeeze(-1)
