import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def reset_parameters(self):
        d_out, d_in = self.weight.shape
        variance = 2.0 / (d_in + d_out)
        std = math.sqrt(variance)
        nn.init.trunc_normal_(
            tensor=self.weight,
            mean=0,
            std=std,
            a=-3*std,
            b=3*std,
        )
    