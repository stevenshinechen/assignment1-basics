import torch
from cs336_basics.optimizer import SGD

def run_sgd_example(num_iterations: int = 100, lr: float = 1):
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)

    for t in range(num_iterations):
        opt.zero_grad() # Reset the gradients for all learnable parameters
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients
        opt.step()      # Run optimizer step

if __name__ == "__main__":
    lrs = [1, 1e1, 1e2, 1e3]
    num_iterations = 10
    for lr in lrs:
        print(f"Running SGD example with {lr=}")
        run_sgd_example(num_iterations=num_iterations, lr=lr)
