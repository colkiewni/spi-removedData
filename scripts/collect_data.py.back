"""
Collect ISMCTS self-play experience data.

Usage:
  python scripts/collect_data.py --shards 500 --sims 25
  python scripts/collect_data.py --shards 500 --sims 25 --resume

Runs on CPU — no GPU needed.
Backs up to Google Drive every SHARD_BACKUP_EVERY shards.
"""

import argparse
import os
import pickle
import random
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    COLLECTION_SIMS, EXP_PER_SHARD,
    SHARD_BACKUP_EVERY, DATA_DIR, DRIVE_DIR
)
from game.engine import (
    GameState, apply_bid, apply_play, PHASE_BID
)
from game.observation import build_observation
from game.bots.ismcts_bot import ISMCTSBot
from game.cards import deal


def collect_game(bots, hands, seed=0):
    state = GameState(
        hands=list(hands), dealer=0,
        player_to_act=0, target_score=200,
    )
    state.trick_leader = 0
    rng       = random.Random(seed)
    decisions = []

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
                decisions.append((p, 'BID', obs, a))
                apply_bid(state, p, a)
            else:
                a = bots[p].play(obs)
                if a not in obs.legal_actions:
                    a = obs.legal_actions[0]
                decisions.append((p, 'PLAY', obs, a))
                apply_play(state, p, a, rng)
        except Exception:
            break

    winner = state.winner
    return [
        {'obs': obs, 'action': action,
         'phase': phase,
         'won': 1 if winner == (p % 2) else 0}
        for (p, phase, obs, action) in decisions
    ]


def backup_to_drive(out_dir, drive_dir):
    if not os.path.exists('/content/drive'):
        return 0
    os.makedirs(drive_dir, exist_ok=True)
    copied = 0
    for fname in os.listdir(out_dir):
        if not fname.endswith('.pkl'):
            continue
        src = os.path.join(out_dir, fname)
        dst = os.path.join(drive_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
            copied += 1
    return copied


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', type=int, default=500)
    parser.add_argument('--sims',   type=int,
                        default=COLLECTION_SIMS)
    parser.add_argument('--outdir', type=str,
                        default=os.path.join(
                            DATA_DIR, 'ismcts_experiences'
                        ))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    drive_dir = os.path.join(
        DRIVE_DIR, 'data', os.path.basename(args.outdir)
    )

    existing    = sorted([
        f for f in os.listdir(args.outdir)
        if f.endswith('.pkl')
    ])
    start_shard = len(existing) if args.resume else 0

    if not args.resume and existing:
        print(f"WARNING: {len(existing)} existing shards. "
              f"Use --resume to continue or delete first.")
        sys.exit(1)

    print(f"Collecting {args.shards} shards")
    print(f"  Sims/bot:  {args.sims}")
    print(f"  Output:    {args.outdir}")
    print(f"  Starting:  shard {start_shard}\n")

    rng  = random.Random(42 + start_shard)
    bots = [ISMCTSBot(n_simulations=args.sims, seed=i)
            for i in range(4)]

    # Estimate speed
    t0    = time.time()
    hands = deal(rng)
    exps  = collect_game(bots, hands, seed=0)
    t1    = time.time()
    spg   = t1 - t0
    epg   = len(exps)
    games = (args.shards * EXP_PER_SHARD) // epg
    print(f"Speed: {epg} exp/game | {spg:.1f}s/game | "
          f"~{games * spg / 3600:.1f}h estimated\n")

    shard_num  = start_shard
    buffer     = list(exps)
    game_count = 1
    t_start    = time.time()

    while shard_num < start_shard + args.shards:
        hands = deal(rng)
        exps  = collect_game(
            bots, hands, seed=42 + game_count
        )
        buffer.extend(exps)
        game_count += 1

        while (len(buffer) >= EXP_PER_SHARD
               and shard_num < start_shard + args.shards):
            path = os.path.join(
                args.outdir,
                f'shard_{shard_num:06d}.pkl'
            )
            with open(path, 'wb') as f:
                pickle.dump(buffer[:EXP_PER_SHARD], f)
            buffer    = buffer[EXP_PER_SHARD:]
            shard_num += 1

            done = shard_num - start_shard
            if done % 10 == 0:
                el  = time.time() - t_start
                rem = ((args.shards - done)
                       / max(done / max(el, 1), 0.001))
                print(f"  Shard {shard_num:4d} | "
                      f"Games {game_count:5d} | "
                      f"{done/args.shards:.0%} | "
                      f"~{rem/60:.0f}m left")

            if shard_num % SHARD_BACKUP_EVERY == 0:
                n = backup_to_drive(args.outdir, drive_dir)
                print(f"  → Drive: {n} shards backed up")

    el = time.time() - t_start
    print(f"\nDone. {shard_num - start_shard} shards "
          f"in {el/60:.0f}m")
    backup_to_drive(args.outdir, drive_dir)
    print("Final backup complete.")


if __name__ == '__main__':
    main()