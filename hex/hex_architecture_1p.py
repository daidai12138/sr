import torch
import torch.nn as nn
from timm import create_model
from musk import utils,modeling
from typing import Optional


def _find_tokens(obj):
    """递归从 hook 的 out 里找第一个 3D tensor: (B, N, D)."""
    if torch.is_tensor(obj):
        return obj if obj.dim() == 3 else None
    if isinstance(obj, (tuple, list)):
        for x in obj:
            t = _find_tokens(x)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for x in obj.values():
            t = _find_tokens(x)
            if t is not None:
                return t
    return None


class TokenToGrid(nn.Module):
    """把 (B, N, dim_in) tokens 映射到 (B, H, W, out_dim)."""

    def __init__(
        self,
        dim_in: int = 1024,
        out_dim: int = 500,
        grid_h: int = 10,
        grid_w: int = 10,
        num_heads: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_queries = grid_h * grid_w
        self.dim_in = dim_in
        self.out_dim = out_dim

        # (1, H*W, dim_in) 的可学习 query
        self.query = nn.Parameter(torch.randn(1, self.num_queries, dim_in) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    dict(
                        ln_q=nn.LayerNorm(dim_in),
                        ln_kv=nn.LayerNorm(dim_in),
                        attn=nn.MultiheadAttention(
                            embed_dim=dim_in,
                            num_heads=num_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        ln_ff=nn.LayerNorm(dim_in),
                        ff=nn.Sequential(
                            nn.Linear(dim_in, dim_in * mlp_ratio),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim_in * mlp_ratio, dim_in),
                            nn.Dropout(dropout),
                        ),
                    )
                )
            )

        self.out_ln = nn.LayerNorm(dim_in)
        self.proj = nn.Identity() if out_dim == dim_in else nn.Linear(dim_in, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N, dim_in) -> (B, H, W, out_dim)"""
        B, N, D = tokens.shape
        if D != self.dim_in:
            raise RuntimeError(f"Token dim mismatch: got {D}, expected {self.dim_in}")

        q = self.query.expand(B, -1, -1)  # (B, H*W, dim_in)
        kv = tokens  # (B, N, dim_in)

        for layer in self.layers:
            q_norm = layer["ln_q"](q)
            kv_norm = layer["ln_kv"](kv)

            attn_out, _ = layer["attn"](q_norm, kv_norm, kv_norm, need_weights=False)
            q = q + attn_out
            q = q + layer["ff"](layer["ln_ff"](q))

        q = self.out_ln(q)  # (B, H*W, dim_in)
        q = self.proj(q)  # (B, H*W, out_dim)
        grid = q.view(B, self.grid_h, self.grid_w, self.out_dim)
        return grid


class SEBlock(nn.Module):
    """常见的通道注意力：Squeeze-and-Excitation（channels-last: (B,H,W,C)）。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise RuntimeError(f"SEBlock expects (B,H,W,C), got {tuple(x.shape)}")
        B, H, W, C = x.shape
        s = x.mean(dim=(1, 2))  # (B, C)
        w = self.fc2(self.act(self.fc1(s)))  # (B, C)
        w = self.gate(w).view(B, 1, 1, C)
        return x * w


class CustomModel(nn.Module):
    def __init__(
        self,
        ckpt_path: str = "./MUSK/model.safetensors",
        model_config: str = "musk_large_patch16_384",
        vocab_size: int = 64010,
        grid_h: int = 10,
        grid_w: int = 10,
        token_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        molan_n: int = 500,
        ca_reduction: int = 16,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.molan_n = molan_n

        # Backbone
        model_musk = create_model(model_config, vocab_size=vocab_size)
        utils.load_model_and_may_interpolate(ckpt_path, model_musk, "model|module", "")
        self.visual = model_musk

        # hook 抓 tokens
        self._last_tokens: Optional[torch.Tensor] = None

        def _save_tokens_hook(module, inp, out):
            t = _find_tokens(out)
            if t is not None:
                self._last_tokens = t

        self.visual.beit3.encoder.register_forward_hook(_save_tokens_hook)

        # tokens -> (B, grid_h, grid_w, molan_n)
        self.token_to_grid = TokenToGrid(
            dim_in=token_dim,
            out_dim=molan_n,
            grid_h=grid_h,
            grid_w=grid_w,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        fused_c = molan_n * 2


        self.out_head = nn.Sequential(
            SEBlock(fused_c, reduction=ca_reduction),
            SEBlock(fused_c, reduction=ca_reduction),
            nn.LayerNorm(fused_c),
            nn.Linear(fused_c, molan_n),
        )

    def forward(self, img: torch.Tensor, point: Optional[torch.Tensor] = None, return_tokens: bool = False):
        """img: (B,3,384,384); point: (B,1,molan_n) or (B,molan_n)."""
        out = self.visual(image=img, with_head=False, out_norm=False)
        global_emb = out[0]

        tokens = self._last_tokens
        if tokens is None:
            raise RuntimeError("没有从 hook 捕获到 tokens (B,N,token_dim)。请确认 hook 挂在正确模块上。")

        grid_feat = self.token_to_grid(tokens)  # (B, grid_h, grid_w, molan_n)

        # point -> (B, 1, molan_n)
        if point is None:
            point = torch.zeros(
                (grid_feat.shape[0], 1, self.molan_n),
                device=grid_feat.device,
                dtype=grid_feat.dtype,
            )
        if point.dim() == 2:
            point = point.unsqueeze(1)
        if point.dim() != 3:
            raise RuntimeError(f"point should be (B,1,C) or (B,C), got {tuple(point.shape)}")
        if point.shape[-1] != self.molan_n:
            raise RuntimeError(f"point dim mismatch: got {point.shape[-1]}, expected {self.molan_n}")

        B = grid_feat.shape[0]
        point_map = point.view(B, 1, 1, self.molan_n).expand(B, self.grid_h, self.grid_w, self.molan_n)

        fused = torch.cat([grid_feat, point_map], dim=-1)  # (B, grid_h, grid_w, 2*molan_n)
        #fused = self.ca1(fused)
        #fused = self.ca2(fused)

        pred = self.out_head(fused)  # (B, grid_h, grid_w, molan_n)

        if return_tokens:
            return pred, tokens, global_emb
        return pred
