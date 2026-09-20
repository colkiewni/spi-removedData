# spades/legal.py
"""
100% deterministic legal-move engine.
The AI never decides legality — this module does.
"""

from .cards import suit_of, suit_mask, cards_of, SPADES, count


def legal_bids(player: int, bids: list, nil_flags: list,
               hand_mask: int, team_scores: list,
               team_bags: list, target_score: int) -> list:
    """
    Return list of legal bid values for this player.
    Normal: 1-13.  Nil: 0.  Blind nil: not allowed.
    """
    # All bids 0-13 are always legal in this ruleset.
    # (Blind nil is explicitly excluded.)
    return list(range(14))  # 0=nil, 1-13=normal


def legal_plays(hand: int, current_trick: list,
                spades_broken: bool) -> list:
    """
    Given the player's hand bitmask and the current trick,
    return a list of legal card ints the player may play.

    current_trick: list of (player, card) tuples already played
                   this trick.  Empty = player is leading.
    """
    if not current_trick:
        # ── Leading ──────────────────────────────────────────
        non_spade = hand & ~_SPADE_MASK
        if not spades_broken:
            if non_spade:
                # Must lead a non-spade
                return cards_of(non_spade)
            else:
                # Only spades remain — allowed to lead spades
                return cards_of(hand)
        else:
            # Spades broken — any card is fine
            return cards_of(hand)
    else:
        # ── Following ────────────────────────────────────────
        led_card = current_trick[0][1]
        led_suit = suit_of(led_card)
        same_suit = suit_mask(hand, led_suit)
        if same_suit:
            # Must follow suit
            return cards_of(same_suit)
        else:
            # Can play anything
            return cards_of(hand)


# Precompute spade mask for speed
from .cards import SUIT_MASKS
_SPADE_MASK = SUIT_MASKS[SPADES]
