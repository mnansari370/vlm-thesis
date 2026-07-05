"""
the CHEAP question-conditioned selector (the student).

A small cross-attention head that scores each visual token by question-relevance using
ONLY pre-LLM information:
  - visual embeds  [nvis, vis_dim]  (Qwen ViT + merger output; no decoder)
  - question embeds [Lq, txt_dim]   (token embeddings; a lookup)

It is trained to predict the mid-layer (L16) TEACHER importance map. At inference it adds
negligible FLOPs (a few M params) and needs NO decoder forward — that is the whole point:
recover the teacher's selection quality without paying for 16+ LLM layers.

Single-sample (bs=1) forward to match the per-sample QwenPruner harness.
"""

import torch
import torch.nn as nn


class CheapQCSelector(nn.Module):
    def __init__(self, vis_dim: int = 3584, txt_dim: int = 3584, d: int = 512,
                 n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vis_in = nn.Sequential(nn.Linear(vis_dim, d), nn.LayerNorm(d))
        self.txt_in = nn.Sequential(nn.Linear(txt_dim, d), nn.LayerNorm(d))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross": nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True),
                "n1": nn.LayerNorm(d),
                "ff": nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(2 * d, d)),
                "n2": nn.LayerNorm(d),
            })
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, vis: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """vis [nvis, vis_dim], q [Lq, txt_dim]  ->  scores [nvis]."""
        v = self.vis_in(vis).unsqueeze(0)          # [1, nvis, d]
        t = self.txt_in(q).unsqueeze(0)            # [1, Lq, d]
        for L in self.layers:
            a, _ = L["cross"](v, t, t)             # visual queries attend to question
            v = L["n1"](v + a)
            v = L["n2"](v + L["ff"](v))
        return self.head(v).squeeze(0).squeeze(-1)  # [nvis]

    @staticmethod
    def num_params(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())
