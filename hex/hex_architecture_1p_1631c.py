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
    """把 (B, N, dim_in) tokens 通过 cross-attn 映射到 (B, H, W, out_dim)."""

    def __init__(
        self,
        dim_in: int = 1024,
        out_dim: int = 1024,       # ✅ 方案2：让它输出 1024（不要压到3）
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
        self.pos_q = nn.Parameter(torch.randn(1, self.num_queries, dim_in) * 0.02)
        # (1, H*W, dim_in) learnable queries
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
        """tokens: (B, N, dim_in) -> grid: (B, H, W, out_dim)"""
        B, N, D = tokens.shape
        if D != self.dim_in:
            raise RuntimeError(f"Token dim mismatch: got {D}, expected {self.dim_in}")
        q = self.query.expand(B, -1, -1) + self.pos_q.expand(B, -1, -1)

        #q = self.query.expand(B, -1, -1)  # (B, H*W, dim_in)
        kv = tokens  # (B, N, dim_in)

        for layer in self.layers:
            q_norm = layer["ln_q"](q)
            kv_norm = layer["ln_kv"](kv)
            attn_out, _ = layer["attn"](q_norm, kv_norm, kv_norm, need_weights=False)
            q = q + attn_out
            q = q + layer["ff"](layer["ln_ff"](q))

        q = self.out_ln(q)          # (B, H*W, dim_in)
        q = self.proj(q)            # (B, H*W, out_dim)
        grid = q.view(B, self.grid_h, self.grid_w, self.out_dim)
        return grid


class CustomModel(nn.Module):
    def __init__(
        self,
        ckpt_path: str = "./MUSK/model.safetensors",
        model_config: str = "musk_large_patch16_384",
        vocab_size: int = 64010,
        grid_h: int = 10,
        grid_w: int = 10,
        token_dim: int = 1024,      # token维度
        num_heads: int = 16,
        num_layers: int = 1,
        dropout: float = 0.1,
        molan_n: int = 3,           # point维度 & 输出维度
        midle_n: int =1024,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.molan_n = molan_n
        self.token_dim = token_dim
        self.midle_n = midle_n

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

        # ✅ 方案2：tokens -> grid_feat (B,10,10,1024)
        self.token_to_grid = TokenToGrid(
            dim_in=token_dim,
            out_dim=midle_n,          # 关键：输出 1024，不要输出3
            grid_h=grid_h,
            grid_w=grid_w,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # ✅ FiLM: point(3) -> gamma/beta(1024)
        self.film = nn.Sequential(
            nn.Linear(molan_n+2, 2048),
            nn.GELU(),
            nn.Linear(2048, 2 * midle_n),
        )

        # ✅ 输出头：1024 -> 3
        # self.out_head = nn.Sequential(
        #     nn.LayerNorm(token_dim),
        #     nn.Linear(token_dim, 256),
        #     nn.GELU(),
        #     #nn.Dropout(0.1),
        #     nn.Linear(256, token_dim),
        # )
        self.out_head_2 = nn.Sequential(
            nn.LayerNorm(midle_n),
            nn.Linear(midle_n, 512),
            #nn.GELU(),
            #nn.Dropout(0.1),
            nn.Linear(512, 1024),
            #nn.Linear(1024, 1024),
            nn.Linear(1024, molan_n),
        )



    def forward(
        self,
        img: torch.Tensor,
        point: Optional[torch.Tensor] = None,
        return_tokens: bool = False
    ):
        """
        img:   (B,3,384,384)
        point: (B,1,3) or (B,3)
        return: pred (B,10,10,3)
        """
        out = self.visual(image=img, with_head=False, out_norm=False)
        global_emb = out[0]

        tokens = self._last_tokens
        if tokens is None:
            raise RuntimeError("没有从 hook 捕获到 tokens (B,N,token_dim)。请确认 hook 挂在正确模块上。")

        grid_feat = self.token_to_grid(tokens)  # (B,H,W,1024)

        # point 处理
        B = grid_feat.shape[0]
        H,W = self.grid_h, self.grid_w
        if point is None:
            point = torch.zeros((B, self.molan_n), device=grid_feat.device, dtype=grid_feat.dtype)

        if point.dim() == 3:  # (B,1,3)
            point = point.squeeze(1)
        if point.dim() != 2:
            raise RuntimeError(f"point should be (B,1,C) or (B,C), got {tuple(point.shape)}")
        if point.shape[-1] != self.molan_n:
            raise RuntimeError(f"point dim mismatch: got {point.shape[-1]}, expected {self.molan_n}")

        # 生成每个像素位置的坐标 (H,W,2)，比如归一化到 [0,1]
        ys = torch.linspace(0, 1, H, device=grid_feat.device)
        xs = torch.linspace(0, 1, W, device=grid_feat.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coord_grid = torch.stack([xx, yy], dim=-1)            # (H,W,2)
        coord_grid = coord_grid.unsqueeze(0).expand(B, H, W, 2)  # (B,H,W,2)
        p_grid = point[:, None, None, :].expand(B, H, W, point.shape[-1])
        film_in = torch.cat([p_grid, coord_grid], dim=-1)  
        gb = self.film(film_in)           # (B,H,W,2*midle_n)

        gamma, beta = gb.chunk(2, dim=-1) # (B,H,W,midle_n)

        fused = grid_feat * (1.0 + gamma) + beta  # (B,H,W,midle_n)
        
        #pred =  fused+pred          # (B,H,W,3) 
        pred = self.out_head_2(fused)

        if return_tokens:
            return pred, tokens, global_emb
        return pred
