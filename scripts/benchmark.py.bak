"""
Benchmark bots against each other.

Usage:
  python scripts/benchmark.py --games 20
  python scripts/benchmark.py --games 20 --neural models/policy_best.pt
  python scripts/benchmark.py --games 20 --neural models/policy_best.pt --neural-only
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import TARGET_SCORE
from game.engine import (
    GameState, apply_bid, apply_play, PHASE_BID
)
from game.observation import build_observation
from game.bots.heuristic_bot import HeuristicBot
from game.bots.ismcts_bot import ISMCTSBot
from game.cards import deal


def run_game(bots, hands, seed=0):
    state = GameState(
        hands=list(hands), dealer=0,
        player_to_act=0, target_score=TARGET_SCORE,
    )
    state.trick_leader = 0
    rng = random.Random(seed)
    while not state.game_over:
        p   = state.player_to_act
        obs = build_observation(state, p)
        if not obs.legal_actions:
            break
        try:
            if state.phase == PHASE_BID:
                a = bots[p].bid(obs)
                if a not in obs.legal_actions:
                    a = obs.legal_actions[0]
                apply_bid(state, p, a)
            else:
                a = bots[p].play(obs)
                if a not in obs.legal_actions:
                    a = obs.legal_actions[0]
                apply_play(state, p, a, rng)
        except Exception:
            break
    return state.winner, list(state.team_scores)


def run_matchup(bots_a, bots_b, n_games,
                label_a, label_b, seed=42):
    rng  = random.Random(seed)
    wins = [0, 0]
    t0   = time.time()
    for i in range(n_games):
        hands     = deal(rng)
        bots      = [bots_a[0], bots_b[0],
                     bots_a[1], bots_b[1]]
        gt0       = time.time()
        winner, _ = run_game(bots, hands, seed=seed + i)
        gt        = time.time() - gt0
        if winner is not None:
            wins[winner] += 1
        el  = time.time() - t0
        rem = ((n_games - i - 1)
               / max((i + 1) / el, 0.001))
        pct = wins[0] / (i + 1)
        print(f"  Game {i+1:3d}/{n_games} | "
              f"{label_a}: {wins[0]} "
              f"{label_b}: {wins[1]} | "
              f"WR: {pct:.1%} | "
              f"{gt:.1f}s/game | "
              f"~{rem:.0f}s left",
              flush=True)
    el  = time.time() - t0
    pct = wins[0] / n_games
    print(f"  RESULT {label_a}: {wins[0]}/{n_games} "
          f"({pct:.1%}) in {el:.0f}s")
    return wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games',       type=int,  default=20)
    parser.add_argument('--neural',      type=str,  default=None)
    parser.add_argument('--sims',        type=int,  default=50)
    parser.add_argument('--neural-only', action='store_true',
                        help='skip baselines, neural vs heuristic only')
    args = parser.parse_args()

    print("=" * 55)
    print("SPADES AI BENCHMARK")
    print("=" * 55)

    h_a = HeuristicBot(seed=1)
    h_b = HeuristicBot(seed=3)

    if not args.neural_only:
        i50a  = ISMCTSBot(n_simulations=50,  seed=1)
        i50b  = ISMCTSBot(n_simulations=50,  seed=3)
        i100a = ISMCTSBot(n_simulations=100, seed=1)
        i100b = ISMCTSBot(n_simulations=100, seed=3)

        print(f"\n[Baseline] ISMCTS-50 vs Heuristic "
              f"({args.games} games)")
        run_matchup([i50a, i50b], [h_a, h_b],
                    args.games, "ISMCTS-50", "Heuristic")

        print(f"\n[Baseline] ISMCTS-100 vs Heuristic "
              f"({args.games} games)")
        run_matchup([i100a, i100b], [h_a, h_b],
                    args.games, "ISMCTS-100", "Heuristic")

    if args.neural:
        print(f"\nLoading neural bot: {args.neural}")
        from game.bots.neural_bot import NeuralBot
        n_a = NeuralBot(args.neural,
                        n_simulations=args.sims,
                        seed=0, name=f'Neural-{args.sims}')
        n_b = NeuralBot(args.neural,
                        n_simulations=args.sims,
                        seed=2, name=f'Neural-{args.sims}')

        print(f"\n[Neural] Neural-{args.sims} vs Heuristic "
              f"({args.games} games)")
        run_matchup([n_a, n_b], [h_a, h_b],
                    args.games,
                    f"Neural-{args.sims}", "Heuristic")

    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)


if __name__ == '__main__':
    main()
