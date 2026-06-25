import torch
import torch.nn as nn
from timm import create_model
from musk import utils, modeling


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
    """
    把 (B, N, dim_in) tokens 通过 learned queries + cross-attn 映射到 (B, H*W, dim_in),
    然后再投影到 out_dim，并 reshape 为 (B, H, W, out_dim).
    """
    def __init__(
        self,
        dim_in: int = 1024,        # 输入 token 维度（MUSK 为 1024）
        out_dim: int = 500,        # 你要的最终维度 molan_n，默认 500
        grid_h: int = 10,
        grid_w: int = 10,
        num_heads: int = 16,
        num_layers: int = 2,
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
                            batch_first=True,  # (B, L, D)
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

        # 最终通道投影：1024 -> out_dim(默认500)
        # 若 out_dim==dim_in，可以直接用 Identity
        self.proj = nn.Identity() if out_dim == dim_in else nn.Linear(dim_in, out_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, dim_in)
        return: (B, grid_h, grid_w, out_dim)
        """
        B, N, D = tokens.shape
        if D != self.dim_in:
            raise RuntimeError(f"Token dim mismatch: got {D}, expected {self.dim_in}")

        q = self.query.expand(B, -1, -1)  # (B, H*W, dim_in)
        kv = tokens                         # (B, N, dim_in)

        for layer in self.layers:
            q_norm = layer["ln_q"](q)
            kv_norm = layer["ln_kv"](kv)

            attn_out, _ = layer["attn"](q_norm, kv_norm, kv_norm, need_weights=False)
            q = q + attn_out

            q = q + layer["ff"](layer["ln_ff"](q))

        q = self.out_ln(q)          # (B, H*W, dim_in)
        q = self.proj(q)            # (B, H*W, out_dim)

        grid = q.view(B, self.grid_h, self.grid_w, self.out_dim)  # (B, H, W, out_dim)
        return grid


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
    ):
        super().__init__()

        # Backbone
        model_musk = create_model(model_config, vocab_size=vocab_size)
        utils.load_model_and_may_interpolate(ckpt_path, model_musk, "model|module", "")
        self.visual = model_musk

        # hook 抓 tokens
        self._last_tokens = None

        def _save_tokens_hook(module, inp, out):
            t = _find_tokens(out)
            if t is not None:
                self._last_tokens = t

        # encoder 输出通常是 (B, N, 1024)
        self.visual.beit3.encoder.register_forward_hook(_save_tokens_hook)

        # 下游：tokens -> (B, 10, 10, 1024)
        self.token_to_grid = TokenToGrid(
            dim_in=1024,    
            out_dim=molan_n,     # molan_n
            grid_h=10,
            grid_w=10,
            num_heads=16,
            num_layers=2,
            dropout=0.1
        )


    def forward(self, img, return_tokens: bool = False):
        # 触发 backbone forward（同时 hook 会把 tokens 存到 self._last_tokens）
        out = self.visual(image=img, with_head=False, out_norm=False)
        global_emb = out[0]  # (B, 1024)，你要的话也可以用

        tokens = self._last_tokens
        if tokens is None:
            raise RuntimeError("没有从 hook 捕获到 tokens (B,N,1024)。请确认 hook 挂在正确模块上。")

        grid_feat = self.token_to_grid(tokens)  # (B,10,10,1024)

        if return_tokens:
            return grid_feat, tokens, global_emb

        return grid_feat

