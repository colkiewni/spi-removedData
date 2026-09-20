# spades/bots/heuristic_bot.py
"""
Heuristic bot — Phase 1 baseline.
Uses hand evaluation for bidding and basic card logic for play.
"""

import random
from .base import BaseBot
from ..observation import PlayerObservation
from ..cards import (suit_of, rank_of, SPADES, CLUBS, DIAMONDS, HEARTS,
                     cards_of, suit_mask, count)


# Rank constants
RANK_ACE   = 12
RANK_KING  = 11
RANK_QUEEN = 10
RANK_JACK  = 9
RANK_TEN   = 8
RANK_NINE  = 7
RANK_EIGHT = 6
RANK_SEVEN = 5


class HeuristicBot(BaseBot):

    def __init__(self, seed=None):
        super().__init__("HeuristicBot")
        self.rng = random.Random(seed)

    # ── Bidding ──────────────────────────────────────────────

    def bid(self, obs: PlayerObservation) -> int:
        hand     = obs.own_hand
        estimate = self._estimate_tricks(hand, obs)

        my_team  = obs.player % 2
        my_score = obs.team_scores[my_team]
        op_score = obs.team_scores[1 - my_team]
        my_bags  = obs.team_bags[my_team]
        target   = obs.target_score

        # Near bag penalty — be conservative
        if my_bags >= 7:
            estimate = max(1, estimate - 1)

        # Near target — be conservative to avoid bags
        if my_score >= target - 30:
            estimate = max(1, estimate - 1)

        # Far behind — be slightly aggressive
        if op_score - my_score > 100:
            estimate = min(13, estimate + 1)

        # Nil check
        if self._should_nil(hand, obs):
            return 0

        return max(1, min(13, estimate))

    def _estimate_tricks(self, hand: int, obs: PlayerObservation) -> int:
        """
        Estimate likely tricks from hand features.
        """
        tricks = 0.0

        for suit in range(4):
            sm = suit_mask(hand, suit)
            cards_in_suit = cards_of(sm)
            n = len(cards_in_suit)
            if n == 0:
                continue

            ranks = sorted([rank_of(c) for c in cards_in_suit], reverse=True)

            if suit == SPADES:
                # Spades are trump — more valuable
                for i, r in enumerate(ranks):
                    if r == RANK_ACE:
                        tricks += 1.0
                    elif r == RANK_KING:
                        tricks += 0.85
                    elif r == RANK_QUEEN:
                        tricks += 0.65
                    elif r == RANK_JACK:
                        tricks += 0.45
                    elif r >= RANK_TEN:
                        tricks += 0.25
                    else:
                        tricks += 0.05
                # Long spade length bonus
                if n >= 5:
                    tricks += 0.5
                if n >= 6:
                    tricks += 0.5
            else:
                # Side suits
                for i, r in enumerate(ranks):
                    if r == RANK_ACE:
                        tricks += 0.95
                    elif r == RANK_KING:
                        # Protected king
                        tricks += 0.70 if n >= 2 else 0.35
                    elif r == RANK_QUEEN:
                        tricks += 0.45 if n >= 3 else 0.20
                    else:
                        break  # Lower cards rarely win

                # Singleton bonus (can ruff)
                if n == 1:
                    tricks += 0.2

        return max(1, round(tricks))

    def _should_nil(self, hand: int, obs: PlayerObservation) -> bool:
        """
        Nil heuristic: consider nil if hand is very weak.
        """
        high_count = 0
        for c in cards_of(hand):
            r = rank_of(c)
            s = suit_of(c)
            if s == SPADES:
                if r >= RANK_QUEEN:
                    high_count += 2
                elif r >= RANK_NINE:
                    high_count += 1
            else:
                if r == RANK_ACE:
                    high_count += 2
                elif r == RANK_KING:
                    high_count += 1

        # Only nil if very weak hand and score situation warrants it
        my_team  = obs.player % 2
        my_score = obs.team_scores[my_team]
        op_score = obs.team_scores[1 - my_team]

        # Don't nil if ahead and comfortable
        if my_score > op_score + 50:
            return False

        return high_count <= 2

    # ── Play ─────────────────────────────────────────────────

    def play(self, obs: PlayerObservation) -> int:
        legal = obs.legal_actions

        if not legal:
            raise ValueError(
                f"No legal actions for player {obs.player}"
            )

        trick = obs.current_trick

        if not trick:
            return self._lead(obs, legal)
        else:
            return self._follow(obs, legal)

    def _lead(self, obs: PlayerObservation, legal: list) -> int:
        """Strategy for leading a trick."""
        # Guard — protect against empty list in rollouts
        if not legal:
            legal = obs.legal_actions
        if not legal:
            raise ValueError(
                f"No legal actions for player {obs.player}"
            )

        p       = obs.player
        partner = (p + 2) % 4

        # If nil — lead lowest non-spade
        if obs.nil_flags[p]:
            return self._lowest_safe(legal, obs)

        # If partner has nil — lead low to protect them
        if obs.nil_flags[partner]:
            return self._lowest_safe(legal, obs)

        # Lead the best winner we have
        return self._lead_winner(obs, legal)

    def _follow(self, obs: PlayerObservation, legal: list) -> int:
        """Strategy for following a trick."""
        p       = obs.player
        partner = (p + 2) % 4
        trick   = obs.current_trick

        led_card = trick[0][1]
        led_suit = suit_of(led_card)

        # Current winning card
        from ..scoring import trick_winner
        current_winner_p = trick_winner(led_suit, trick)
        partner_winning  = (current_winner_p == partner)

        # Nil protection — play lowest possible
        if obs.nil_flags[p]:
            return min(legal, key=lambda c: rank_of(c))

        # Partner winning and partner has nil — duck
        if partner_winning and obs.nil_flags[partner]:
            return min(legal, key=lambda c: rank_of(c))

        # Can follow suit?
        can_follow = any(suit_of(c) == led_suit for c in legal)

        if can_follow:
            same_suit_cards = [
                c for c in legal if suit_of(c) == led_suit
            ]

            if partner_winning:
                # Partner winning — play low, don't steal the trick
                return min(
                    same_suit_cards, key=lambda c: rank_of(c)
                )
            else:
                # Try to win with lowest winning card
                winners = [
                    c for c in same_suit_cards
                    if self._beats_trick(c, trick)
                ]
                if winners:
                    return min(winners, key=lambda c: rank_of(c))
                else:
                    # Can't win — dump lowest
                    return min(
                        same_suit_cards, key=lambda c: rank_of(c)
                    )
        else:
            # Can't follow suit — ruff or discard
            spades = [c for c in legal if suit_of(c) == SPADES]

            if spades and not partner_winning:
                # Ruff with lowest spade
                return min(spades, key=lambda c: rank_of(c))
            else:
                # Discard lowest non-spade
                non_spades = [
                    c for c in legal if suit_of(c) != SPADES
                ]
                if non_spades:
                    return min(
                        non_spades, key=lambda c: rank_of(c)
                    )
                # Only spades — play lowest
                return min(legal, key=lambda c: rank_of(c))

    def _lead_winner(self, obs: PlayerObservation,
                     legal: list) -> int:
        """Lead the best winner we have."""
        if not legal:
            # Fallback to first available legal action
            if obs.legal_actions:
                return obs.legal_actions[0]
            raise ValueError(
                f"No legal cards at all for player {obs.player}"
            )

        # Prefer aces in non-spade suits
        for c in legal:
            if rank_of(c) == RANK_ACE and suit_of(c) != SPADES:
                return c

        # Then kings in non-spade suits
        for c in legal:
            if rank_of(c) == RANK_KING and suit_of(c) != SPADES:
                return c

        # Then spade ace
        for c in legal:
            if rank_of(c) == RANK_ACE and suit_of(c) == SPADES:
                return c

        # Otherwise lead lowest non-spade
        non_spades = [c for c in legal if suit_of(c) != SPADES]
        if non_spades:
            return min(non_spades, key=lambda c: rank_of(c))

        # Only spades left
        return min(legal, key=lambda c: rank_of(c))

    def _lowest_safe(self, legal: list,
                     obs: PlayerObservation) -> int:
        """
        Return the lowest card, preferring non-spades.
        Used for nil play and nil protection.
        """
        non_spades = [c for c in legal if suit_of(c) != SPADES]
        if non_spades:
            return min(non_spades, key=lambda c: rank_of(c))
        return min(legal, key=lambda c: rank_of(c))

    def _beats_trick(self, card: int, trick: list) -> bool:
        """
        Check if playing this card would currently win the trick.
        Uses sentinel player 99 to identify the test card.
        """
        if not trick:
            return True
        led_suit = suit_of(trick[0][1])
        from ..scoring import trick_winner
        test_trick = list(trick) + [(99, card)]
        return trick_winner(led_suit, test_trick) == 99
