"""
Policy network — predicts action probabilities from game state.
Input:  (batch, FEATURE_DIM)
Output: (batch, 52) logits, one per card
"""

import torch
import torch.nn as nn
from config import FEATURE_DIM, HIDDEN_DIM, N_LAYERS


class PolicyNet(nn.Module):
    def __init__(self,
                 input_dim  = FEATURE_DIM,
                 hidden_dim = HIDDEN_DIM,
                 n_layers   = N_LAYERS):
        super().__init__()
        layers = []
        d      = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(d, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ]
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Linear(hidden_dim, 52)

    def forward(self, x, legal_mask=None):
        """
        Args:
            x:           (batch, input_dim)
            legal_mask:  (batch, 52) bool — True for legal actions
                         If None, all actions are considered legal

        Returns:
            logits: (batch, 52)
                    Illegal actions are set to -1e9 if mask provided
        """
        logits = self.head(self.trunk(x))
        if legal_mask is not None:
            # legal_mask must be (batch, 52) — not (52,)
            logits = logits.masked_fill(~legal_mask, -1e9)
        return logits


def load_policy(path, device='cpu'):
    """
    Load a saved PolicyNet checkpoint.
    Handles both raw state dict and wrapped dict formats.
    """
    model = PolicyNet().to(device)
    ckpt  = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters())