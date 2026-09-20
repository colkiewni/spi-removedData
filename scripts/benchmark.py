# scripts/benchmark.py
"""
Benchmark bots against each other.

Usage:
  python scripts/benchmark.py --games 20
  python scripts/benchmark.py --games 20 --neural models/policy_best.pt
  python scripts/benchmark.py --games 20 --neural models/policy_best.pt --neural-only
  python scripts/benchmark.py --games 200 --workers 12
"""

import argparse
import os
import random
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import TARGET_SCORE
from game.engine import GameState, apply_bid, apply_play, PHASE_BID
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


def _run_one(args):
    """Worker: run a single game. Args are picklable config, not bot objects."""
    bot_type_a, bot_type_b, neural_path, sims, seed, game_idx = args

    rng   = random.Random(seed + game_idx)
    hands = deal(rng)

    def make_bot(spec):
        if spec == 'heuristic':
            return HeuristicBot(seed=seed + game_idx)
        if spec == 'ismcts50':
            return ISMCTSBot(n_simulations=50,  seed=seed + game_idx)
        if spec == 'ismcts100':
            return ISMCTSBot(n_simulations=100, seed=seed + game_idx)
        if spec == 'neural':
            from game.bots.neural_bot import NeuralBot
            return NeuralBot(neural_path, n_simulations=sims,
                             seed=seed + game_idx, name=f'Neural-{sims}')

    bots = [make_bot(bot_type_a), make_bot(bot_type_b),
            make_bot(bot_type_a), make_bot(bot_type_b)]
    winner, scores = run_game(bots, hands, seed=seed + game_idx)
    return game_idx, winner, scores


def run_matchup_parallel(bot_type_a, bot_type_b, n_games,
                         label_a, label_b,
                         neural_path=None, sims=50,
                         seed=42, workers=1):
    job_args = [(bot_type_a, bot_type_b, neural_path, sims, seed, i)
                for i in range(n_games)]

    wins = [0, 0]
    results = {}
    t0 = time.time()

    with Pool(processes=workers) as pool:
        for game_idx, winner, scores in pool.imap_unordered(_run_one, job_args):
            results[game_idx] = (winner, scores)
            if winner is not None:
                wins[winner] += 1
            done = len(results)
            el   = time.time() - t0
            rem  = ((n_games - done) / max(done / el, 0.001))
            pct  = wins[0] / done
            print(f"  Game {done:3d}/{n_games} | "
                  f"{label_a}: {wins[0]} "
                  f"{label_b}: {wins[1]} | "
                  f"WR: {pct:.1%} | "
                  f"~{rem:.0f}s left",
                  flush=True)

    el  = time.time() - t0
    pct = wins[0] / n_games
    print(f"  RESULT {label_a}: {wins[0]}/{n_games} ({pct:.1%}) in {el:.0f}s")
    return wins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games',       type=int, default=20)
    parser.add_argument('--neural',      type=str, default=None)
    parser.add_argument('--sims',        type=int, default=50)
    parser.add_argument('--neural-only', action='store_true')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--workers',     type=int, default=1)
    args = parser.parse_args()

    print("=" * 55)
    print("SPADES AI BENCHMARK")
    print("=" * 55)

    kw = dict(seed=args.seed, workers=args.workers)

    if not args.neural_only:
        print(f"\n[Baseline] ISMCTS-50 vs Heuristic ({args.games} games)")
        run_matchup_parallel('ismcts50', 'heuristic', args.games,
                             "ISMCTS-50", "Heuristic", **kw)

        print(f"\n[Baseline] ISMCTS-100 vs Heuristic ({args.games} games)")
        run_matchup_parallel('ismcts100', 'heuristic', args.games,
                             "ISMCTS-100", "Heuristic", **kw)

    if args.neural:
        print(f"\nLoading neural bot: {args.neural}")
        print(f"\n[Neural] Neural-{args.sims} vs Heuristic ({args.games} games)")
        run_matchup_parallel('neural', 'heuristic', args.games,
                             f"Neural-{args.sims}", "Heuristic",
                             neural_path=args.neural, sims=args.sims, **kw)

    print("\n" + "=" * 55)
    print("DONE")
    print("=" * 55)


if __name__ == '__main__':
    main()
