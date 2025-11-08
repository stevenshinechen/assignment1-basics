from collections.abc import Callable, Iterable
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.Tensor | dict], lr: float = 1e-3):
        assert lr >= 0
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)                  # Iteration number
                grad = p.grad.data                     # Gradient of loss wrt p
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place
                state["t"] = t + 1

        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor | dict],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                grad = p.grad.data
                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * (grad**2)

                t = state["t"]
                lr_t = lr * (math.sqrt(1 - beta2**t) / (1 - beta1**t))
                
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                p.data -= lr * weight_decay * p.data
                state["t"] += 1

        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters) / (cosine_cycle_iters - warmup_iters))) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate
