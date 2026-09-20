# spades/bots/ismcts_bot.py
"""
Information Set Monte Carlo Tree Search bot.

This is the first serious AI bot (Phase 2).
It handles hidden information by sampling possible worlds,
running MCTS on each, and aggregating action values.
"""

import math
import random
import time
from collections import defaultdict
from typing import List, Dict, Optional

from .base import BaseBot
from .heuristic_bot import HeuristicBot
from ..observation import PlayerObservation
from ..engine import (
    clone, apply_bid, apply_play,
    get_legal_actions, PHASE_BID, PHASE_PLAY, PHASE_DONE
)
from ..cards import (
    cards_of, suit_of, rank_of, count, SPADES,
    SUIT_MASKS, mask_of
)


class ISMCTSNode:
    """Node in the ISMCTS tree."""

    __slots__ = ["visits", "wins", "children", "untried"]

    def __init__(self, untried: list):
        self.visits   = 0
        self.wins     = 0.0
        self.children: Dict[int, "ISMCTSNode"] = {}
        self.untried   = list(untried)

    def uct_select(self, legal: list, c: float = 1.4) -> int:
        """Select action using UCT formula."""
        best_action = None
        best_score  = -float("inf")

        for action in legal:
            if action not in self.children:
                return action

            child = self.children[action]
            if child.visits == 0:
                return action

            exploit = child.wins / child.visits
            explore = c * math.sqrt(
                math.log(self.visits) / child.visits
            )
            score = exploit + explore

            if score > best_score:
                best_score  = score
                best_action = action

        return best_action


class ISMCTSBot(BaseBot):
    """
    ISMCTS bot with configurable simulation budget.

    n_simulations: number of determinizations to sample
    time_limit:    seconds budget (overrides n_simulations)
    """

    def __init__(self,
                 n_simulations: int = 500,
                 time_limit: float = None,
                 seed: int = None,
                 name: str = "ISMCTSBot"):
        super().__init__(name)
        self.n_simulations = n_simulations
        self.time_limit    = time_limit
        self.rng           = random.Random(seed)
        self._rollout_bots = [
            HeuristicBot(seed=(seed or 0) + i) for i in range(4)
        ]

    # ── Public interface ─────────────────────────────────────

    def bid(self, obs: PlayerObservation) -> int:
        """Use ISMCTS to choose the best bid."""
        return self._ismcts_action(obs, is_bid=True)

    def play(self, obs: PlayerObservation) -> int:
        """Use ISMCTS to choose the best card to play."""
        return self._ismcts_action(obs, is_bid=False)

    # ── Core ISMCTS ──────────────────────────────────────────

    def _ismcts_action(self, obs: PlayerObservation,
                       is_bid: bool) -> int:
        """
        Run ISMCTS and return the best action.
        """
        legal = obs.legal_actions
        if not legal:
            raise ValueError(
                f"No legal actions for player {obs.player}"
            )
        if len(legal) == 1:
            return legal[0]

        # Action visit/win counts across all determinizations
        action_visits = defaultdict(int)
        action_wins   = defaultdict(float)

        start     = time.time()
        sim_count = 0

        while True:
            # Check budget
            if self.time_limit is not None:
                if time.time() - start >= self.time_limit:
                    break
            else:
                if sim_count >= self.n_simulations:
                    break

            # Sample a determinization (possible world)
            det_state = self._determinize(obs)
            if det_state is None:
                sim_count += 1
                continue

            # Run one simulation on this determinization
            action, value = self._run_simulation(
                det_state, obs.player, legal
            )

            action_visits[action] += 1
            action_wins[action]   += value
            sim_count += 1

        # Choose action with highest average value
        if not action_visits:
            return self.rng.choice(legal)

        best_action = max(
            legal,
            key=lambda a: (
                action_wins[a] / action_visits[a]
                if action_visits[a] > 0 else -1.0
            )
        )
        return best_action

    def _run_simulation(self, state, player: int,
                        legal: list):
        """
        Run one simulation on a determinized state.
        Returns (action_taken, value).
        """
        # Pick a random action from legal moves
        action = self.rng.choice(legal)

        # Clone and apply action
        sim = clone(state)
        try:
            if sim.phase == PHASE_BID:
                apply_bid(sim, sim.player_to_act, action)
            else:
                apply_play(sim, sim.player_to_act, action)
        except (AssertionError, ValueError):
            return action, 0.0

        # Rollout to completion
        value = self._rollout(sim, player, max_tricks=13)
        return action, value

    def _rollout(self, state, player: int,
                 max_tricks: int = 13) -> float:
        """
        Rollout using heuristic bots.
        Returns value in [0,1] from perspective of player's team.
        """
        my_team = player % 2

        for _ in range(max_tricks * 4 + 20):
            if state.game_over or state.phase == PHASE_DONE:
                break

            p   = state.player_to_act
            obs = self._fast_obs(state, p)

            # Skip if no legal actions
            if not obs.legal_actions:
                break

            try:
                if state.phase == PHASE_BID:
                    action = self._rollout_bots[p].bid(obs)
                    if action not in obs.legal_actions:
                        action = obs.legal_actions[0]
                    apply_bid(state, p, action)
                else:
                    action = self._rollout_bots[p].play(obs)
                    if action not in obs.legal_actions:
                        action = obs.legal_actions[0]
                    apply_play(state, p, action)
            except (AssertionError, ValueError, IndexError):
                break

        return self._evaluate(state, my_team)

    def _evaluate(self, state, my_team: int) -> float:
        """
        Return a value in [0, 1] for my_team.
        Uses score difference normalized by target.
        """
        s0 = state.team_scores[my_team]
        s1 = state.team_scores[1 - my_team]

        if state.game_over:
            if state.winner == my_team:
                return 1.0
            elif state.winner is not None:
                return 0.0

        # Normalize score difference
        diff  = s0 - s1
        scale = state.target_score
        return max(0.0, min(1.0, 0.5 + diff / (2.0 * scale)))

    # ── Determinization ──────────────────────────────────────

    def _determinize(self, obs: PlayerObservation):
        """
        Create a plausible complete game state consistent
        with the player's observation.

        Samples unknown cards (opponent/partner hands)
        while respecting all known constraints:
          - Own hand is fixed
          - Already played cards are excluded
          - Void information is respected
        """
        from ..engine import GameState

        p = obs.player

        # Cards not in our hand and not yet played
        all_cards_mask = (1 << 52) - 1
        unknown_mask   = (
            (~obs.own_hand) & (~obs.played_cards) & all_cards_mask
        )
        unknown_cards  = cards_of(unknown_mask)

        # Other players who need cards assigned
        other_players = [q for q in range(4) if q != p]

        # How many cards should each player have?
        # Each player starts with 13, loses one per trick won,
        # and may have already played in current trick.
        cards_in_trick = {q: 0 for q in range(4)}
        for (tp, tc) in obs.current_trick:
            cards_in_trick[tp] = 1

        target_counts = {}
        for q in other_players:
            target_counts[q] = max(
                0,
                13 - obs.tricks_won[q] - cards_in_trick[q]
            )

        total_needed = sum(target_counts.values())

        # If mismatch, redistribute evenly
        if total_needed != len(unknown_cards):
            available = len(unknown_cards)
            per       = available // 3
            rem       = available % 3
            for i, q in enumerate(other_players):
                target_counts[q] = per + (1 if i < rem else 0)

        # Build possible card sets per player respecting voids
        possible = {}
        for q in other_players:
            possible[q] = [
                c for c in unknown_cards
                if not obs.known_voids[q][suit_of(c)]
            ]

        # Shuffle unknown cards for random assignment
        shuffled = list(unknown_cards)
        self.rng.shuffle(shuffled)

        assigned = {q: [] for q in range(4)}
        assigned[p] = cards_of(obs.own_hand)

        remaining = list(shuffled)

        for q in other_players:
            need  = target_counts[q]
            valid = [c for c in remaining if c in possible[q]]
            self.rng.shuffle(valid)

            chosen = valid[:need]

            # If not enough valid cards, fill with any remaining
            if len(chosen) < need:
                extras = [
                    c for c in remaining if c not in chosen
                ]
                self.rng.shuffle(extras)
                chosen += extras[:need - len(chosen)]

            assigned[q] = chosen[:need]
            chosen_set  = set(assigned[q])
            remaining   = [c for c in remaining
                           if c not in chosen_set]

        # Build hands as bitmasks
        hands = [mask_of(assigned[q]) for q in range(4)]

        # Construct determinized GameState
        state = GameState(
            hands         = hands,
            played_cards  = obs.played_cards,
            played_by     = list(obs.played_by),
            dealer        = obs.dealer,
            player_to_act = obs.player,
            phase         = obs.phase,
            bids          = list(obs.bids),
            nil_flags     = list(obs.nil_flags),
            current_trick = list(obs.current_trick),
            trick_leader  = obs.trick_leader,
            tricks_won    = list(obs.tricks_won),
            known_voids   = [list(v) for v in obs.known_voids],
            spades_broken = obs.spades_broken,
            team_scores   = list(obs.team_scores),
            team_bags     = list(obs.team_bags),
            hand_number   = obs.hand_number,
            target_score  = obs.target_score,
        )

        # Compute tricks_played from tricks_won
        state.tricks_played = sum(obs.tricks_won)

        return state

    def _fast_obs(self, state, player: int):
        """Build observation for rollout bot."""
        from ..observation import build_observation
        return build_observation(state, player)
