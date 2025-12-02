import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    starts = np.random.randint(0, n - context_length, size=batch_size)

    offsets = np.arange(context_length)
    idx = starts[:, None] + offsets

    inputs = torch.tensor(dataset[idx], device=device)
    targets = torch.tensor(dataset[idx+1], device=device)
    return inputs, targets
