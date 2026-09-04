"""Tiny CPU-only example: train a 2-layer MLP on synthetic regression data
using the gradient-transform wrapper, with both SGD and AdamW.
"""

import torch
import torch.nn as nn

from grad_transform import signed_square, step_with_transformed_gradients


def make_synthetic_data(n: int = 64, in_dim: int = 4, out_dim: int = 2, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, in_dim, generator=g)
    true_w = torch.randn(in_dim, out_dim, generator=g)
    y = x @ true_w + 0.01 * torch.randn(n, out_dim, generator=g)
    return x, y


def make_model(in_dim: int = 4, out_dim: int = 2) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, out_dim))


def train(optimizer_cls, optimizer_kwargs, steps: int = 20) -> float:
    x, y = make_synthetic_data()
    model = make_model()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    loss = None
    for _ in range(steps):
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        step_with_transformed_gradients(optimizer, signed_square)

    return loss.item()


if __name__ == "__main__":
    final_loss_sgd = train(torch.optim.SGD, {"lr": 0.05, "momentum": 0.9})
    print(f"SGD   final loss: {final_loss_sgd:.6f}")

    final_loss_adamw = train(torch.optim.AdamW, {"lr": 1e-2})
    print(f"AdamW final loss: {final_loss_adamw:.6f}")
