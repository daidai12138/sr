"""
Model definition (MUSK visual backbone + regression head).

Optimizations vs original:
- forward() no longer takes unused `labels`
- configurable head dims / dropout / activation
- helper methods to freeze/unfreeze backbone
"""
from __future__ import annotations

import torch
import torch.nn as nn
from timm import create_model
from musk import utils , modeling


class CustomModel(nn.Module):
    def __init__(
        self,
        visual_output_dim: int,
        num_outputs: int,
        model_config: str = "musk_large_patch16_384",
        ckpt_path: str = "./MUSK/model.safetensors",
        head_dims: tuple[int, int] = (2048, 4096),
        dropout: float = 0.3,
        activation: str = "gelu",
    ):
        super().__init__()

        # ---- backbone ----
        model_musk = create_model(model_config, vocab_size=64010)
        utils.load_model_and_may_interpolate(ckpt_path, model_musk, "model|module", "")
        self.visual = model_musk

        # ---- head ----
        act: nn.Module
        if activation.lower() == "relu":
            act = nn.ReLU()
        else:
            act = nn.GELU()

        d1, d2 = head_dims
        self.regression_head = nn.Sequential(
            nn.Linear(visual_output_dim, d1),
            act,
            #nn.Dropout(p=dropout),
            nn.Linear(d1, d2),
            act,
            #nn.Dropout(p=dropout),
        )
        self.regression_head1 = nn.Linear(d2, num_outputs)

    @torch.no_grad()
    def freeze_backbone(self):
        for p in self.visual.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def unfreeze_backbone(self):
        for p in self.visual.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = True):
        # MUSK returns a tuple/list; first element is pooled embedding
        feats = self.visual(image=x, with_head=False, out_norm=False)[0]
        h = self.regression_head(feats)
        preds = self.regression_head1(h)
        return (preds, h) if return_features else preds
