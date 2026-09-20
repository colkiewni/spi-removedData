# spades/bots/base.py
"""Abstract base class for all bots."""

from abc import ABC, abstractmethod
from ..observation import PlayerObservation


class BaseBot(ABC):
    """
    All bots inherit from this.
    They receive only a PlayerObservation — never the true GameState.
    """

    def __init__(self, name: str = "Bot"):
        self.name = name

    @abstractmethod
    def bid(self, obs: PlayerObservation) -> int:
        """Return a bid value (0=nil, 1-13=normal)."""
        pass

    @abstractmethod
    def play(self, obs: PlayerObservation) -> int:
        """Return a card integer to play."""
        pass

    def reset(self):
        """Called at the start of each new game."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"
