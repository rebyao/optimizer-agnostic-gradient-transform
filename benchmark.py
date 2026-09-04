"""Lightweight CPU benchmark: baseline optimizer.step() vs.
step_with_transformed_gradients(optimizer, signed_square).

Not a rigorous microbenchmark - just a rough sense of wrapper overhead.
"""

import time

import torch
import torch.nn as nn

from grad_transform import signed_square, step_with_transformed_gradients


def build_model(hidden: int = 256) -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(128, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    )


def run(label: str, use_transform: bool, steps: int = 200, warmup: int = 20) -> float:
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(64, 128)
    y = torch.randn(64, 10)

    def step():
        model.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.mse_loss(out, y)
        loss.backward()
        if use_transform:
            step_with_transformed_gradients(optimizer, signed_square)
        else:
            optimizer.step()

    for _ in range(warmup):
        step()

    start = time.perf_counter()
    for _ in range(steps):
        step()
    elapsed = time.perf_counter() - start

    avg_ms = elapsed / steps * 1000
    print(f"{label:>24}: {avg_ms:.4f} ms/step  ({steps} steps)")
    return avg_ms


if __name__ == "__main__":
    baseline_ms = run("baseline (no transform)", use_transform=False)
    transformed_ms = run("transformed (wrapper)", use_transform=True)

    overhead_ms = transformed_ms - baseline_ms
    overhead_pct = (overhead_ms / baseline_ms) * 100
    print(f"\nWrapper overhead: {overhead_ms:.4f} ms/step ({overhead_pct:.1f}%)")
