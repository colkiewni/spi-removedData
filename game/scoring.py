# spades/scoring.py
"""
Completely isolated scoring engine.
All game-result logic lives here.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class HandResult:
    score_delta: List[int]       # [team0_delta, team1_delta]
    new_scores:  List[int]       # [team0_score, team1_score]
    new_bags:    List[int]       # [team0_bags,  team1_bags]
    game_over:   bool
    winner:      Optional[int]   # 0, 1, or None
    boston:      List[bool]      # [team0_boston, team1_boston]
    nil_results: List[bool]      # [p0_nil_ok, p1_nil_ok, p2_nil_ok, p3_nil_ok]


def calculate_hand_score(
    bids:         List[int],
    nil_flags:    List[bool],
    tricks_won:   List[int],
    team_scores:  List[int],
    team_bags:    List[int],
    target_score: int = 200,
    negative_cutoff: int = -200,
) -> HandResult:
    """
    Compute end-of-hand scoring.
    Teams: 0 = players 0&2,  1 = players 1&3
    """
    # ── Nil results ──────────────────────────────────────────
    nil_results = [False, False, False, False]
    for p in range(4):
        if nil_flags[p]:
            nil_results[p] = (tricks_won[p] == 0)

    # ── Per-team totals ──────────────────────────────────────
    score_delta = [0, 0]
    new_bags    = [team_bags[0], team_bags[1]]

    for team in range(2):
        p0 = team        # players on this team
        p1 = team + 2

        team_tricks = tricks_won[p0] + tricks_won[p1]

        # ── Nil scoring ──────────────────────────────────────
        for p in [p0, p1]:
            if nil_flags[p]:
                if nil_results[p]:
                    score_delta[team] += 100
                else:
                    score_delta[team] -= 100

        # ── Normal contract ──────────────────────────────────
        # Only non-nil bids count toward the contract target
        normal_bid = sum(
            bids[p] for p in [p0, p1] if not nil_flags[p]
        )

        # Tricks that count toward contract =
        # ONLY the non-nil player's tricks
        # (nil player's tricks do NOT help make the contract
        #  but a failed nil's tricks still count as bags
        #  against the non-nil player's contract)
        contract_tricks = sum(
            tricks_won[p] for p in [p0, p1] if not nil_flags[p]
        )

        # Boston check: team bid exactly 13 (no nils) and won all 13
        is_boston = (
            normal_bid == 13
            and not any(nil_flags[p] for p in [p0, p1])
            and team_tricks == 13
        )

        if is_boston:
            score_delta[team] += 200
        elif normal_bid > 0:
            if contract_tricks >= normal_bid:
                # Made contract
                bags = contract_tricks - normal_bid
                score_delta[team] += 10 * normal_bid + bags
                new_bags[team] += bags
            else:
                # Set (failed contract)
                score_delta[team] -= 10 * normal_bid

        # ── Bag penalty ──────────────────────────────────────
        old_penalty_count = team_bags[team] // 10
        new_penalty_count = new_bags[team] // 10
        if new_penalty_count > old_penalty_count:
            score_delta[team] -= 100 * (
                new_penalty_count - old_penalty_count
            )

    # ── Apply deltas ─────────────────────────────────────────
    new_scores = [
        team_scores[0] + score_delta[0],
        team_scores[1] + score_delta[1],
    ]

    # ── Boston flag (for reporting) ──────────────────────────
    boston = [False, False]
    for team in range(2):
        p0, p1 = team, team + 2
        normal_bid = sum(bids[p] for p in [p0, p1] if not nil_flags[p])
        team_tricks = tricks_won[p0] + tricks_won[p1]
        boston[team] = (
            normal_bid == 13
            and not any(nil_flags[p] for p in [p0, p1])
            and team_tricks == 13
        )

    # ── Game-over check ──────────────────────────────────────
    game_over = False
    winner    = None

    # Negative cutoff
    for t in range(2):
        if new_scores[t] <= negative_cutoff:
            game_over = True
            winner = 1 - t

    # Target reached
    if not game_over:
        t0_done = new_scores[0] >= target_score
        t1_done = new_scores[1] >= target_score
        if t0_done or t1_done:
            game_over = True
            if new_scores[0] > new_scores[1]:
                winner = 0
            elif new_scores[1] > new_scores[0]:
                winner = 1
            else:
                game_over = False

    return HandResult(
        score_delta=score_delta,
        new_scores=new_scores,
        new_bags=new_bags,
        game_over=game_over,
        winner=winner,
        boston=boston,
        nil_results=nil_results,
    )

def trick_winner(led_suit: int, plays: list) -> int:
    """
    Determine which player won the trick.

    plays: list of (player, card) in play order.
    led_suit: suit index of the led card.

    Returns player index of winner.
    """
    from .cards import suit_of, rank_of, SPADES

    best_player = plays[0][0]
    best_card   = plays[0][1]
    best_suit   = suit_of(best_card)
    best_rank   = rank_of(best_card)

    for player, card in plays[1:]:
        card_suit = suit_of(card)
        card_rank = rank_of(card)

        if card_suit == best_suit:
            # Same suit — higher rank wins
            if card_rank > best_rank:
                best_player = player
                best_card   = card
                best_suit   = card_suit
                best_rank   = card_rank
        elif card_suit == SPADES and best_suit != SPADES:
            # Spade beats non-spade
            best_player = player
            best_card   = card
            best_suit   = card_suit
            best_rank   = card_rank
        # Otherwise card cannot beat current best

    return best_player
