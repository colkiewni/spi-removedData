# SPI

Spades card game AI using Information Set Monte Carlo Tree Search (ISMCTS)
guided by a neural network policy trained on self-play data.

## Architecture
- **Game engine**: Full 4-player Spades with bidding, nil bids, bags
- **ISMCTS**: Handles hidden information via determinization
- **Policy net**: 4-layer MLP trained on ISMCTS self-play moves
- **Neural bot**: ISMCTS guided by policy priors (PUCT selection)

## Benchmark results
| Bot | vs Heuristic | vs ISMCTS-100 |
|-----|-------------|---------------|
| Heuristic | baseline | — |
| ISMCTS-50 | 60% | — |
| ISMCTS-100 | 70% | baseline |
| ISMCTS-200 | 86% | — |
| NeuralBot-50 | TBD | TBD |

## Usage

### Collect training data (CPU)
```bash
python scripts/collect_data.py --shards 500 --sims 25

Train policy network (GPU recommended)
python scripts/train_policy.py --data data/ismcts_experiences

Run benchmark
# CPU only
python scripts/benchmark.py --games 50

# With neural bot (GPU)
python scripts/benchmark.py --games 50 --neural models/policy_best.pt

Hardware

    Data collection: CPU only
    Training: GPU strongly recommended (T4 on Colab)
    Inference: CPU works, GPU faster



Project structure
game/           game engine, bots
neural/         feature extraction, policy network
scripts/        data collection, training, benchmarks
data/           self-play experience shards (gitignored)
models/         trained model checkpoints (gitignored)
config.py       all constants and hyperparameters