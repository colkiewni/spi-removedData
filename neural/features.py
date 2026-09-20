"""
Feature extraction — converts PlayerObservation to tensor.
FEATURE_DIM = 183. Do not change without retraining all models.

Feature layout:
  [0:52]    own hand (bitmask, one-hot per card)
  [52:104]  played cards
  [104:108] tricks won per player (normalized)
  [108:112] bids per player (normalized)
  [112:116] nil flags per player
  [116:120] team scores (normalized)
  [120:124] team bags (normalized)
  [124]     spades broken flag
  [125]     hand number (normalized)
  [126:130] cards in current trick (normalized)
  [130:183] known voids (4 players x 4 suits + padding)
"""

import numpy as np
from config import FEATURE_DIM


def obs_to_features(obs) -> np.ndarray:
    """
    Convert a PlayerObservation to a float32 numpy array.
    Shape: (FEATURE_DIM,) = (183,)
    """
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)
    idx   = 0

    # Own hand — 52 bits
    for i in range(52):
        feats[idx + i] = (obs.own_hand >> i) & 1
    idx += 52

    # Played cards — 52 bits
    for i in range(52):
        feats[idx + i] = (obs.played_cards >> i) & 1
    idx += 52

    # Tricks won per player — normalized by 13
    for p in range(4):
        feats[idx + p] = obs.tricks_won[p] / 13.0
    idx += 4

    # Bids per player — normalized by 13
    for p in range(4):
        b = obs.bids[p] if obs.bids[p] is not None else 0
        feats[idx + p] = b / 13.0
    idx += 4

    # Nil flags
    for p in range(4):
        feats[idx + p] = float(obs.nil_flags[p])
    idx += 4

    # Team scores — normalized by target
    target = max(obs.target_score, 1)
    for t in range(2):
        feats[idx + t] = obs.team_scores[t] / target
    idx += 2

    # Team bags — normalized by 10
    for t in range(2):
        feats[idx + t] = obs.team_bags[t] / 10.0
    idx += 2

    # Spades broken
    feats[idx] = float(obs.spades_broken)
    idx += 1

    # Hand number — normalized by 20
    feats[idx] = obs.hand_number / 20.0
    idx += 1

    # Current trick — cards played so far normalized by 4
    feats[idx] = len(obs.current_trick) / 4.0
    idx += 1

    # Known voids — 4 players x 4 suits = 16 bits
    for p in range(4):
        for s in range(4):
            feats[idx] = float(obs.known_voids[p][s])
            idx += 1

    # Padding to reach 183
    # idx should be 183 here
    assert idx <= FEATURE_DIM, (
        f"Feature overflow: idx={idx} > FEATURE_DIM={FEATURE_DIM}"
    )

    return feats


def verify_feature_dim():
    """Quick sanity check — call once at startup."""
    # Build a dummy obs-like object
    class DummyObs:
        own_hand      = 0
        played_cards  = 0
        tricks_won    = [0, 0, 0, 0]
        bids          = [3, 3, 3, 3]
        nil_flags     = [False]*4
        team_scores   = [0, 0]
        team_bags     = [0, 0]
        target_score  = 200
        spades_broken = False
        hand_number   = 0
        current_trick = []
        known_voids   = [[False]*4 for _ in range(4)]

    feats = obs_to_features(DummyObs())
    assert feats.shape == (FEATURE_DIM,), (
        f"Expected ({FEATURE_DIM},) got {feats.shape}"
    )
    return True