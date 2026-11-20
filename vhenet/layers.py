"""Full-window virus encoder and controlled host branch for ViH-GAT.

The encoder consumes every 1024 bp sliding window of a viral genome
(CLS + token-mean + token-max; 7,680 dimensions per window) rather than a
sparse subset of windows. Windows are projected, passed through residual 1-D
convolutions along the window axis, and pooled with masked mean and max
operations. Host information enters the score through an embedding, a
similarity projection, and explicit interaction terms. An optional auxiliary
host head can absorb host-only signal during training; inference uses the main
head only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowResBlock(nn.Module):
    """Residual convolution block along the window axis (order preserving)."""

    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.ln1 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel, padding=pad)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [Nwin, D]
        residual = x
        y = self.conv1(x.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        y = F.gelu(self.ln1(y))
        y = self.dropout(y)
        y = self.conv2(y.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        y = self.ln2(y)
        return F.gelu(y + residual)


def mlp(dims, dropout, final_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        elif final_act:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ViHGAT_814(nn.Module):
    """Full-window virus encoder with an interaction classifier."""

    def __init__(
        self,
        num_hosts: int,
        host_sim_matrix,
        window_feat_dim: int = 7680,  # 3 x 2560 (cls + mean + max)
        embed_dim: int = 256,
        dropout: float = 0.25,
        window_cnn_blocks: int = 3,
        window_cnn_kernel: int = 5,
        max_windows: int = 512,
        use_host_head: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_host_head = use_host_head

        # ---- Virus branch: all windows ----
        self.window_projector = nn.Sequential(
            nn.Linear(window_feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.window_pos = nn.Parameter(torch.zeros(max_windows, embed_dim))
        nn.init.normal_(self.window_pos, mean=0.0, std=0.02)
        self.window_cnn = nn.ModuleList(
            [WindowResBlock(embed_dim, window_cnn_kernel, dropout)
             for _ in range(window_cnn_blocks)]
        )
        self.virus_agg = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ---- Host branch ----
        self.host_embedding = nn.Embedding(num_hosts, embed_dim)
        self.host_mlp = mlp([embed_dim, embed_dim * 2, embed_dim], dropout)
        self.sim_projector = mlp([num_hosts, embed_dim], dropout)
        self.host_fusion = mlp([embed_dim * 2, embed_dim], dropout)

        # ---- Interaction ----
        self.virus_interact = nn.Linear(embed_dim, embed_dim)
        self.host_interact = nn.Linear(embed_dim, embed_dim)
        self.classifier = mlp([embed_dim * 4, 256, 64, 1], dropout)

        # ---- Auxiliary host-prior head (training only; unused at inference) ----
        if use_host_head:
            self.host_head = mlp([embed_dim, 128, 1], dropout)

        if not isinstance(host_sim_matrix, torch.Tensor):
            host_sim_matrix = torch.tensor(host_sim_matrix, dtype=torch.float32)
        if tuple(host_sim_matrix.shape) != (num_hosts, num_hosts):
            raise ValueError(
                f"host_sim_matrix shape {tuple(host_sim_matrix.shape)} "
                f"!= ({num_hosts}, {num_hosts})"
            )
        self.register_buffer("sim_matrix", host_sim_matrix.float())

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.host_embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_virus(self, windows: torch.Tensor, window_mask: torch.Tensor):
        """windows: [Nwin, 7680], mask: [Nwin] -> virus_rep [D]"""
        x = self.window_projector(windows)  # [Nwin, D]
        n = x.shape[0]
        x = x + self.window_pos[:n]
        x = x * window_mask.unsqueeze(-1)
        for block in self.window_cnn:
            x = block(x)
        x = x * window_mask.unsqueeze(-1)
        denom = window_mask.sum().clamp_min(1.0)
        mean_pool = x.sum(dim=0) / denom
        max_pool = x.masked_fill(window_mask.unsqueeze(-1) == 0, -1e4).max(dim=0).values
        return self.virus_agg(torch.cat([mean_pool, max_pool], dim=0))

    def host_rep_for(self, host_ids: torch.Tensor):
        emb = self.host_mlp(self.host_embedding(host_ids))
        sim = self.sim_projector(self.sim_matrix[host_ids])
        return self.host_fusion(torch.cat([emb, sim], dim=-1))

    def forward(self, windows, window_mask, host_ids):
        """windows: [Nwin, 7680]; host_ids: [B] -> logits [B,1]"""
        virus_rep = self.encode_virus(windows, window_mask)  # [D]
        virus_rep = virus_rep.unsqueeze(0).expand(host_ids.shape[0], -1)
        host_rep = self.host_rep_for(host_ids)

        inter = self.virus_interact(virus_rep) * self.host_interact(host_rep)
        diff = virus_rep - host_rep
        feats = torch.cat([virus_rep, host_rep, inter, diff], dim=-1)
        logits = self.classifier(feats)  # [B,1]

        host_logits = None
        if self.use_host_head:
            host_logits = self.host_head(host_rep)  # [B,1]
        return logits, host_logits

    def score_all_hosts(self, windows, window_mask, all_host_ids):
        """Score every host for one virus (inference)."""
        logits, _ = self.forward(windows, window_mask, all_host_ids)
        return logits.squeeze(-1)
