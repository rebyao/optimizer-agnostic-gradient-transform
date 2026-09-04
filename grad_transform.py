"""Optimizer-agnostic gradient transformation mechanism.

Provides a mechanism to temporarily transform every parameter gradient
(after loss.backward(), before optimizer.step()), run a standard PyTorch
optimizer step against the transformed gradients, and then restore the
original gradients afterwards.

The mechanism only relies on the standard PyTorch optimizer interface
(``optimizer.param_groups`` and ``optimizer.step()``), so it works with any
``torch.optim.Optimizer`` subclass without reimplementing optimizer-specific
update rules.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

__all__ = ["signed_square", "step_with_transformed_gradients"]


def signed_square(g: torch.Tensor) -> torch.Tensor:
    """Sign-preserving square: T(g) = sign(g) * g^2 = g * abs(g).

    Note: this is NOT the same as ``g.square()`` (a.k.a. ``g ** 2``), which
    is always non-negative and discards the sign of the gradient.
    """
    return g * g.abs()


def step_with_transformed_gradients(
    optimizer: torch.optim.Optimizer,
    transform_fn: Callable[[torch.Tensor], torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run ``optimizer.step()`` against transformed gradients.

    For every parameter across all ``optimizer.param_groups`` that has a
    non-None ``.grad``, this:

    1. Saves a clone of the original gradient.
    2. Replaces ``p.grad`` in-place with ``transform_fn(original_grad)``.
    3. Calls ``optimizer.step(*args, **kwargs)`` so the optimizer consumes
       the transformed gradients.
    4. Restores every saved original gradient onto ``p.grad``, even if
       ``optimizer.step()`` raised an exception.

    Parameters with ``p.grad is None`` are left untouched (skipped, and no
    gradient is fabricated for them).

    Args:
        optimizer: Any ``torch.optim.Optimizer`` instance.
        transform_fn: Callable mapping an original gradient tensor to a
            transformed gradient tensor, e.g. ``signed_square``.
        *args, **kwargs: Forwarded to ``optimizer.step()``.

    Returns:
        Whatever ``optimizer.step()`` returns (typically ``None``, or a loss
        value if a closure is used).
    """
    saved_grads: list[tuple[torch.nn.Parameter, torch.Tensor]] = []

    try:
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Clone BEFORE any in-place mutation so `original` cannot
                # alias the tensor we are about to transform.
                original = p.grad.clone()
                saved_grads.append((p, original))
                transformed = transform_fn(original)
                # Guard against transform_fn returning the same tensor
                # object it was given (e.g. an identity transform): if we
                # assigned it directly, an in-place optimizer update could
                # silently corrupt `original` through aliasing.
                if transformed is original:
                    transformed = transformed.clone()
                p.grad = transformed

        return optimizer.step(*args, **kwargs)
    finally:
        for p, original in saved_grads:
            p.grad = original
