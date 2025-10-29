import math
import torch
import torch.nn as nn
from einops import einsum

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
