# spades/observation.py
"""
Player observation — what a player is legally allowed to know.
The AI NEVER receives the true GameState directly.
It always receives a PlayerObservation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlayerObservation:
    """
    Everything a player can legally observe.
    No opponent hand information is included.
    """
    # Identity
    player:       int
    dealer:       int
    hand_number:  int
    target_score: int

    # Own hand
    own_hand:     int          # bitmask

    # What has been played (public knowledge)
    played_cards: int          # bitmask of all played cards
    played_by:    List[int]    # played_by[card] = player who played it (-1=unplayed)

    # Bidding
    bids:         List[Optional[int]]   # [p0,p1,p2,p3]  None=not yet bid
    nil_flags:    List[bool]

    # Current trick
    current_trick: List[tuple]   # (player, card) in play order
    trick_leader:  int
    tricks_won:    List[int]     # [p0,p1,p2,p3]

    # Void information (derived from play history)
    # known_voids[player][suit] = True if player is known void in suit
    known_voids:  List[List[bool]]

    # Score
    team_scores:  List[int]
    team_bags:    List[int]

    # State flags
    spades_broken: bool
    phase:         str

    # Legal actions at this moment
    legal_actions: List[int]   # card ints or bid ints


def build_observation(state, player: int) -> "PlayerObservation":
    """
    Extract a PlayerObservation from a GameState for the given player.
    This is the ONLY way the AI should access game state.
    """
    from .cards import cards_of

    legal: List[int]
    if state.phase == "BID":
        from .legal import legal_bids
        legal = legal_bids(
            player, state.bids, state.nil_flags,
            state.hands[player],
            state.team_scores, state.team_bags,
            state.target_score
        )
    else:
        from .legal import legal_plays
        legal = legal_plays(
            state.hands[player],
            state.current_trick,
            state.spades_broken
        )

    return PlayerObservation(
        player        = player,
        dealer        = state.dealer,
        hand_number   = state.hand_number,
        target_score  = state.target_score,
        own_hand      = state.hands[player],
        played_cards  = state.played_cards,
        played_by     = list(state.played_by),
        bids          = list(state.bids),
        nil_flags     = list(state.nil_flags),
        current_trick = list(state.current_trick),
        trick_leader  = state.trick_leader,
        tricks_won    = list(state.tricks_won),
        known_voids   = [list(v) for v in state.known_voids],
        team_scores   = list(state.team_scores),
        team_bags     = list(state.team_bags),
        spades_broken = state.spades_broken,
        phase         = state.phase,
        legal_actions = legal,
    )
