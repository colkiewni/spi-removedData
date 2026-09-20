# spades/engine.py
"""
Deterministic game engine / simulator.
All game logic lives here.
The AI never modifies this state directly.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Optional

from .cards import deal, cards_of, suit_of, count, SPADES
from .legal import legal_plays, legal_bids
from .scoring import calculate_hand_score, trick_winner


PHASE_BID  = "BID"
PHASE_PLAY = "PLAY"
PHASE_DONE = "DONE"


@dataclass
class GameState:
    # ── Card state ───────────────────────────────────────────
    hands:        List[int]           # 4 bitmasks (true state)
    played_cards: int  = 0            # bitmask of all played cards
    played_by:    List[int] = field(  # played_by[card] = player
                      default_factory=lambda: [-1]*52)

    # ── Turn tracking ────────────────────────────────────────
    dealer:         int = 0
    player_to_act:  int = 0
    phase:          str = PHASE_BID

    # ── Bidding ──────────────────────────────────────────────
    bids:       List[Optional[int]] = field(default_factory=lambda: [None]*4)
    nil_flags:  List[bool]          = field(default_factory=lambda: [False]*4)

    # ── Trick state ──────────────────────────────────────────
    current_trick: List[tuple] = field(default_factory=list)  # (player, card)
    trick_leader:  int = 0
    tricks_won:    List[int] = field(default_factory=lambda: [0]*4)
    tricks_played: int = 0

    # ── Void tracking ────────────────────────────────────────
    # known_voids[player][suit] = True if player is known void
    known_voids: List[List[bool]] = field(
        default_factory=lambda: [[False]*4 for _ in range(4)])

    # ── Spades ───────────────────────────────────────────────
    spades_broken: bool = False

    # ── Scoring ──────────────────────────────────────────────
    team_scores: List[int] = field(default_factory=lambda: [0, 0])
    team_bags:   List[int] = field(default_factory=lambda: [0, 0])

    # ── Game tracking ────────────────────────────────────────
    hand_number:  int = 0
    target_score: int = 200
    game_over:    bool = False
    winner:       Optional[int] = None

    # ── History (for training data) ──────────────────────────
    history: list = field(default_factory=list)


def new_game(target_score: int = 200,
             dealer: int = 0,
             rng=None) -> GameState:
    """Create a fresh game and deal the first hand."""
    state = GameState(
        hands        = deal(rng),
        dealer       = dealer,
        player_to_act= dealer,   # dealer bids first
        target_score = target_score,
    )
    state.trick_leader = dealer
    return state


def _start_new_hand(state: GameState, rng=None) -> None:
    """Reset per-hand state and deal new cards."""
    state.hands        = deal(rng)
    state.played_cards = 0
    state.played_by    = [-1] * 52
    state.bids         = [None] * 4
    state.nil_flags    = [False] * 4
    state.current_trick= []
    state.trick_leader = state.dealer
    state.tricks_won   = [0] * 4
    state.tricks_played= 0
    state.known_voids  = [[False]*4 for _ in range(4)]
    state.spades_broken= False
    state.player_to_act= state.dealer
    state.phase        = PHASE_BID
    state.hand_number += 1


def apply_bid(state: GameState, player: int, bid: int) -> None:
    """Apply a bid action to the game state."""
    assert state.phase == PHASE_BID
    assert state.player_to_act == player
    assert bid in legal_bids(
        player, state.bids, state.nil_flags,
        state.hands[player],
        state.team_scores, state.team_bags,
        state.target_score
    )

    state.bids[player]      = bid
    state.nil_flags[player] = (bid == 0)

    # Advance to next bidder or start play
    next_player = (player + 1) % 4
    bids_done   = all(b is not None for b in state.bids)

    if bids_done:
        state.phase         = PHASE_PLAY
        state.player_to_act = state.dealer   # dealer leads first trick
        state.trick_leader  = state.dealer
    else:
        state.player_to_act = next_player


def apply_play(state: GameState, player: int, card: int,
               rng=None) -> None:
    """Apply a card-play action to the game state."""
    assert state.phase == PHASE_PLAY
    assert state.player_to_act == player

    legal = legal_plays(
        state.hands[player],
        state.current_trick,
        state.spades_broken
    )
    assert card in legal, (
        f"Illegal play: {card} not in {legal}"
    )

    # Remove card from hand
    state.hands[player]  &= ~(1 << card)
    state.played_cards   |=  (1 << card)
    state.played_by[card] = player

    # Track spades broken
    if suit_of(card) == SPADES:
        state.spades_broken = True

    # Track voids: if player didn't follow led suit, they're void
    if state.current_trick:
        led_suit = suit_of(state.current_trick[0][1])
        if suit_of(card) != led_suit:
            state.known_voids[player][led_suit] = True

    state.current_trick.append((player, card))

    # Check if trick is complete (4 cards played)
    if len(state.current_trick) == 4:
        _resolve_trick(state, rng)
    else:
        state.player_to_act = (player + 1) % 4


def _resolve_trick(state: GameState, rng=None) -> None:
    """Resolve a completed trick."""
    led_suit = suit_of(state.current_trick[0][1])
    winner   = trick_winner(led_suit, state.current_trick)

    state.tricks_won[winner] += 1
    state.tricks_played      += 1
    state.current_trick       = []
    state.trick_leader        = winner

    # Check if hand is complete (13 tricks)
    if state.tricks_played == 13:
        _resolve_hand(state, rng)
    else:
        state.player_to_act = winner


def _resolve_hand(state: GameState, rng=None) -> None:
    """Score the completed hand and check for game over."""
    result = calculate_hand_score(
        bids         = state.bids,
        nil_flags    = state.nil_flags,
        tricks_won   = state.tricks_won,
        team_scores  = state.team_scores,
        team_bags    = state.team_bags,
        target_score = state.target_score,
    )

    state.team_scores = result.new_scores
    state.team_bags   = result.new_bags

    # Store result in history
    state.history.append({
        "hand":        state.hand_number,
        "bids":        list(state.bids),
        "nil_flags":   list(state.nil_flags),
        "tricks_won":  list(state.tricks_won),
        "score_delta": result.score_delta,
        "new_scores":  result.new_scores,
        "boston":      result.boston,
        "nil_results": result.nil_results,
    })

    if result.game_over:
        state.game_over = True
        state.winner    = result.winner
        state.phase     = PHASE_DONE
    else:
        # Rotate dealer and start new hand
        state.dealer = (state.dealer + 1) % 4
        _start_new_hand(state, rng)


def get_legal_actions(state: GameState) -> list:
    """Return legal actions for the current player."""
    p = state.player_to_act
    if state.phase == PHASE_BID:
        return legal_bids(
            p, state.bids, state.nil_flags,
            state.hands[p],
            state.team_scores, state.team_bags,
            state.target_score
        )
    elif state.phase == PHASE_PLAY:
        return legal_plays(
            state.hands[p],
            state.current_trick,
            state.spades_broken
        )
    return []


def clone(state: GameState) -> GameState:
    """Deep copy of game state for search/simulation."""
    s = GameState(
        hands        = list(state.hands),
        played_cards = state.played_cards,
        played_by    = list(state.played_by),
        dealer       = state.dealer,
        player_to_act= state.player_to_act,
        phase        = state.phase,
        bids         = list(state.bids),
        nil_flags    = list(state.nil_flags),
        current_trick= list(state.current_trick),
        trick_leader = state.trick_leader,
        tricks_won   = list(state.tricks_won),
        tricks_played= state.tricks_played,
        known_voids  = [list(v) for v in state.known_voids],
        spades_broken= state.spades_broken,
        team_scores  = list(state.team_scores),
        team_bags    = list(state.team_bags),
        hand_number  = state.hand_number,
        target_score = state.target_score,
        game_over    = state.game_over,
        winner       = state.winner,
    )
    # Don't copy history for search clones (saves memory)
    return s
