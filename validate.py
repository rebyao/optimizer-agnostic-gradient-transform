"""Deterministic CPU checks against native SGD and AdamW references."""

import torch

from grad_transform import signed_square, step_with_transformed_gradients


def validate(optimizer_cls, **kwargs):
    parameter = torch.nn.Parameter(
        torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    )
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = optimizer_cls([parameter], **kwargs)
    reference_optimizer = optimizer_cls([reference], **kwargs)

    for values in (
        [-2.0, 0.5, 0.0, 1.5],
        [0.25, -3.0, 1.0, -0.5],
        [1.0, 0.0, -0.75, 2.0],
    ):
        original = torch.tensor(values, dtype=torch.float64)
        parameter.grad = original.clone()
        # Independent reference: transform explicitly, then use the native step.
        reference.grad = original * original.abs()
        reference_optimizer.step()
        step_with_transformed_gradients(optimizer, signed_square)

        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, original, rtol=0, atol=0)

    print(f"{optimizer_cls.__name__}: parameter updates and gradient restoration passed")


if __name__ == "__main__":
    validate(torch.optim.SGD, lr=0.1, momentum=0.9)
    validate(torch.optim.AdamW, lr=0.01, weight_decay=0.01)
