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