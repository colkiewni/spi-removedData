"""
Neural-guided ISMCTS bot.
Policy net called ONCE per decision for priors.
Rollouts use heuristic bots — no GPU inside simulation loop.
"""

import math
import random
import os
from collections import defaultdict

import torch
import torch.nn.functional as F

from config import PUCT_C, ROLLOUT_MAX, NEURAL_SIMS
from neural.features import obs_to_features
from neural.policy_net import load_policy
from game.engine import (
    apply_bid, apply_play, clone,
    PHASE_BID, PHASE_DONE
)
from game.observation import build_observation
from game.cards import cards_of, suit_of, mask_of
from game.bots.heuristic_bot import HeuristicBot


class NeuralBot:
    """
    ISMCTS guided by a policy network trained on ISMCTS self-play.

    Key design decisions (learned from failed attempts):
      - Policy net called ONCE per decision, not once per sim
      - No value net for move selection (proven broken)
      - Rollouts use heuristic bots (fast, CPU only)
      - Determinization respects known voids
    """

    def __init__(self,
                 policy_path,
                 n_simulations = NEURAL_SIMS,
                 seed          = None,
                 name          = 'NeuralBot'):
        self.name          = name
        self.n_simulations = n_simulations
        self.rng           = random.Random(seed)
        self.device        = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.policy_net = load_policy(policy_path, self.device)
        self._hbots     = [
            HeuristicBot(seed=(seed or 0) + i)
            for i in range(4)
        ]

    def bid(self, obs):
        return self._act(obs)

    def play(self, obs):
        return self._act(obs)

    # ── Decision making ───────────────────────────────────────

    def _act(self, obs):
        legal = obs.legal_actions
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]

        # ONE policy call for all priors
        priors        = self._get_priors(obs, legal)
        action_visits = defaultdict(int)
        action_wins   = defaultdict(float)

        for sim_i in range(self.n_simulations):
            det = self._determinize(obs)
            if det is None:
                continue
            action = self._puct_select(
                legal, priors, action_visits,
                action_wins, sim_i
            )
            value = self._rollout(det, obs.player, action)
            action_visits[action] += 1
            action_wins[action]   += value

        if not action_visits:
            return max(legal,
                       key=lambda a: priors.get(a, 0.0))

        return max(
            legal,
            key=lambda a: (
                action_wins[a] / action_visits[a]
                if action_visits[a] > 0 else 0.0
            )
        )

    @torch.no_grad()
    def _get_priors(self, obs, legal):
        """Call policy net once, return prior dict."""
        if obs.phase != 'PLAY':
            return {a: 1.0 / len(legal) for a in legal}

        feats  = obs_to_features(obs)
        x      = torch.tensor(
            feats, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        mask   = torch.zeros(
            1, 52, dtype=torch.bool
        ).to(self.device)
        for a in legal:
            if 0 <= a < 52:
                mask[0, a] = True

        logits = self.policy_net(x, mask)[0]
        logits[~mask[0]] = -1e9
        probs  = F.softmax(logits / 0.5, dim=0)
        return {
            a: probs[a].item()
            for a in legal if 0 <= a < 52
        }

    def _puct_select(self, legal, priors,
                     visits, wins, total):
        """PUCT formula for action selection."""
        N      = max(1, total)
        best_a = None
        best_s = -float('inf')
        for a in legal:
            n_a   = visits[a]
            q_a   = wins[a] / n_a if n_a > 0 else 0.5
            p_a   = priors.get(a, 1.0 / len(legal))
            score = (q_a
                     + PUCT_C * p_a * math.sqrt(N) / (1 + n_a))
            if score > best_s:
                best_s, best_a = score, a
        return best_a or self.rng.choice(legal)

    def _rollout(self, state, player, action):
        """Apply action then play out with heuristic bots."""
        sim     = clone(state)
        my_team = player % 2

        try:
            if sim.phase == PHASE_BID:
                apply_bid(sim, sim.player_to_act, action)
            else:
                apply_play(sim, sim.player_to_act, action)
        except Exception:
            return 0.5

        if sim.game_over:
            return 1.0 if sim.winner == my_team else 0.0

        for _ in range(ROLLOUT_MAX):
            if sim.game_over or sim.phase == PHASE_DONE:
                break
            p   = sim.player_to_act
            obs = build_observation(sim, p)
            if not obs.legal_actions:
                break
            try:
                if sim.phase == PHASE_BID:
                    a = self._hbots[p].bid(obs)
                    if a not in obs.legal_actions:
                        a = obs.legal_actions[0]
                    apply_bid(sim, p, a)
                else:
                    a = self._hbots[p].play(obs)
                    if a not in obs.legal_actions:
                        a = obs.legal_actions[0]
                    apply_play(sim, p, a)
            except Exception:
                break

        if sim.game_over:
            return 1.0 if sim.winner == my_team else 0.0

        s0 = sim.team_scores[my_team]
        s1 = sim.team_scores[1 - my_team]
        return max(0.0, min(1.0,
            0.5 + (s0 - s1) / (2 * sim.target_score)
        ))

    def _determinize(self, obs):
        """Sample a consistent world from the observation."""
        p            = obs.player
        all_mask     = (1 << 52) - 1
        unknown_mask = (
            (~obs.own_hand) & (~obs.played_cards) & all_mask
        )
        unknown = cards_of(unknown_mask)
        others  = [q for q in range(4) if q != p]

        in_trick = {q: 0 for q in range(4)}
        for (tp, _) in obs.current_trick:
            in_trick[tp] = 1

        counts = {
            q: max(0, 13 - obs.tricks_won[q] - in_trick[q])
            for q in others
        }
        if sum(counts.values()) != len(unknown):
            av  = len(unknown)
            per = av // 3
            rem = av % 3
            for i, q in enumerate(others):
                counts[q] = per + (1 if i < rem else 0)

        possible = {
            q: [c for c in unknown
                if not obs.known_voids[q][suit_of(c)]]
            for q in others
        }

        shuffled    = list(unknown)
        self.rng.shuffle(shuffled)
        assigned    = {q: [] for q in range(4)}
        assigned[p] = cards_of(obs.own_hand)
        remaining   = list(shuffled)

        for q in others:
            need   = counts[q]
            valid  = [c for c in remaining
                      if c in possible[q]]
            self.rng.shuffle(valid)
            chosen = valid[:need]
            if len(chosen) < need:
                extra = [c for c in remaining
                         if c not in set(chosen)]
                self.rng.shuffle(extra)
                chosen += extra[:need - len(chosen)]
            assigned[q] = chosen[:need]
            used        = set(assigned[q])
            remaining   = [c for c in remaining
                           if c not in used]

        hands = [mask_of(assigned[q]) for q in range(4)]

        try:
            from game.engine import GameState
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
                known_voids   = [list(v)
                                 for v in obs.known_voids],
                spades_broken = obs.spades_broken,
                team_scores   = list(obs.team_scores),
                team_bags     = list(obs.team_bags),
                hand_number   = obs.hand_number,
                target_score  = obs.target_score,
            )
            state.tricks_played = sum(obs.tricks_won)
            return state
        except Exception:
            return None