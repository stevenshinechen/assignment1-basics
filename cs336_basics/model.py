import math
import torch
import torch.nn as nn
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float, Int
from torch import Tensor

def reset_weight(weight: nn.Parameter) -> nn.Parameter:
    d_out, d_in = weight.shape
    variance = 2.0 / (d_in + d_out)
    std = math.sqrt(variance)
    return nn.init.trunc_normal_(
        tensor=weight,
        mean=0,
        std=std,
        a=-3*std,
        b=3*std,
    )

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def reset_parameters(self):
        reset_weight(self.weight)

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

    def reset_parameters(self):
        reset_weight(self.weight)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        rms_norm = self.weight * (x / rms)

        return rms_norm.to(in_dtype)

def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k)) # (d_k/2,)
        positions = torch.arange(max_seq_len, device=device) # (seq_len,)
        radians = einsum(positions, inv_freq, "seq_len, half_d-> seq_len half_d") # (seq_len, d_k/2) outer product
        
        self.register_buffer('cos', torch.cos(radians)) # (seq_len, d_k/2)
        self.register_buffer('sin', torch.sin(radians)) # (seq_len, d_k/2)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_k"]:
        cos = self.cos[token_positions] # (..., seq_len, d_k/2)
        sin = self.sin[token_positions] # (..., seq_len, d_k/2)

        x1 = x[..., ::2]  # (..., seq_len, d_k/2)
        x2 = x[..., 1::2] # (..., seq_len, d_k/2)

        x_rotated_even = x1 * cos - x2 * sin # (..., seq_len, d_k/2)
        x_rotated_odd = x1 * sin + x2 * cos  # (..., seq_len, d_k/2)

        x_stacked = torch.stack([x_rotated_even, x_rotated_odd], dim=-1) # (..., seq_len, d_k/2, 2)
        x_rotated = rearrange(x_stacked, "... seq_len half_d two -> ... seq_len (half_d two)")
        return x_rotated

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_stable = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_stable)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    scale_factor = 1 / math.sqrt(d_k)
    attn_weight = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") * scale_factor

    if mask is not None:
        attn_weight.masked_fill_(mask.logical_not(), -float('inf'))

    attn_weight = softmax(attn_weight, dim=-1)
    attn = einsum(attn_weight, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return attn

class MultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 8192,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.rope = rope

        # d_model == num_heads * self.d_k
        self.W_Q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_K = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_V = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_O = Linear(d_model, d_model, device=device, dtype=dtype)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"],
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = token_positions.shape[-1]

        Q = self.W_Q(x)  # (..., seq_len, d_model)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = rearrange(Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        K = rearrange(K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads)
        V = rearrange(V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads)

        if self.rope is not None:
            token_pos_repeated = repeat(token_positions, "... seq_len -> ... num_heads seq_len", num_heads=self.num_heads)
            Q = self.rope(Q, token_pos_repeated)
            K = self.rope(K, token_pos_repeated)

        mask = self.causal_mask[:seq_len, :seq_len]

        attn_out = scaled_dot_product_attention(Q, K, V, mask=mask)
        multihead_out = rearrange(attn_out, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        return self.W_O(multihead_out)
