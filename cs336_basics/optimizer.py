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
