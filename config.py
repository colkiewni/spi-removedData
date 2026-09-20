"""
Central configuration — all constants live here.
Import this instead of hardcoding values anywhere.
"""

# ── Game ──────────────────────────────────────────────────────
TARGET_SCORE   = 200      # points to win a game
N_PLAYERS      = 4
N_CARDS        = 52
N_SUITS        = 4
CARDS_PER_HAND = 13

# ── Features ──────────────────────────────────────────────────
FEATURE_DIM    = 183      # obs_to_features output size
                           # DO NOT change without retraining

# ── Neural network ────────────────────────────────────────────
HIDDEN_DIM     = 256      # PolicyNet hidden layer size
N_LAYERS       = 4        # PolicyNet number of layers

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE     = 512
VAL_BATCH_SIZE = 1024
LR             = 0.001
WEIGHT_DECAY   = 1e-4
MAX_EPOCHS     = 100
PATIENCE       = 8        # early stopping
VAL_FRACTION   = 0.1
GRAD_CLIP      = 1.0

# ── Data collection ───────────────────────────────────────────
COLLECTION_SIMS  = 25     # ISMCTS sims for data collection
                           # 25 = ~6s/game, fits in Colab session
                           # 100 = ~25s/game, too slow for collection
EXP_PER_SHARD    = 500    # experiences per .pkl shard file
SHARD_BACKUP_EVERY = 50   # backup to cloud every N shards

# ── ISMCTS ────────────────────────────────────────────────────
EVAL_SIMS      = 100      # sims for evaluation benchmarks
NEURAL_SIMS    = 50       # sims for neural bot (target: match EVAL_SIMS strength)
PUCT_C         = 1.5      # exploration constant
ROLLOUT_MAX    = 200      # max steps in a rollout

# ── Storage ───────────────────────────────────────────────────
GITHUB_USER    = "colkiewni"
GITHUB_REPO    = "api"
HF_REPO        = "colbionas/spades-ai"     # set to "username/spades-ai-models" after HF setup

DATA_DIR       = "data"
MODELS_DIR     = "models"
DRIVE_DIR      = "/content/drive/MyDrive/spades_ai"