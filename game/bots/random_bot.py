# spades/bots/random_bot.py
"""Plays and bids randomly (but legally)."""

import random
from .base import BaseBot
from ..observation import PlayerObservation


class RandomBot(BaseBot):

    def __init__(self, seed=None):
        super().__init__("RandomBot")
        self.rng = random.Random(seed)

    def bid(self, obs: PlayerObservation) -> int:
        # Random bid 1-7 (avoid nil/overbid randomly)
        return self.rng.randint(1, 7)

    def play(self, obs: PlayerObservation) -> int:
        return self.rng.choice(obs.legal_actions)
