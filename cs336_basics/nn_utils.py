from typing import Iterable
import torch
from jaxtyping import Float, Int
from torch import Tensor


def log_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_stable = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_stable)
    return x_stable - torch.log(exp_x.sum(dim=dim, keepdim=True))


def cross_entropy(
    inputs: Float[Tensor, "batch vocab_size"],
    targets: Int[Tensor, "batch"]
) -> Float[Tensor, ""]:
    log_probs = log_softmax(inputs, dim=-1)
    batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
    return -log_probs[batch_indices, targets].mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    g_norm = 0
    for p in parameters:
        if p.grad is not None:
            g_norm += p.grad.data.norm(2)**2
    
    g_norm = g_norm.sqrt()

    if g_norm >= max_l2_norm:
        clip_coeff = max_l2_norm / (g_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad *= clip_coeff
