# Optimizer-Agnostic Gradient Transformation

A small mechanism for intercepting parameter gradients between
`loss.backward()` and `optimizer.step()`, applying a transformation to them,
running the optimizer's own `step()` against the transformed gradients, and
then restoring the original gradients afterwards.

## What it does

`step_with_transformed_gradients(optimizer, transform_fn, *args, **kwargs)`:

1. Walks every `optimizer.param_groups[*]["params"]`.
2. Skips any parameter with `p.grad is None`.
3. Clones and saves each parameter's original gradient.
4. Replaces `p.grad` with `transform_fn(original_grad)`.
5. Calls `optimizer.step(*args, **kwargs)` so the optimizer's normal update
   rule runs against the transformed gradients.
6. Restores every original gradient onto `p.grad`, in a `finally` block, so
   restoration happens even if `optimizer.step()` raises.

The included transform is:

```python
def signed_square(g: torch.Tensor) -> torch.Tensor:
    return g * g.abs()  # T(g) = sign(g) * g^2
```

Note this is deliberately *not* `g.square()` / `g ** 2`, which is always
non-negative and would discard the sign of the gradient.

## Why it's optimizer-agnostic

The wrapper never touches optimizer internals (no SGD/AdamW math is
reimplemented). It only depends on the standard `torch.optim.Optimizer`
interface:

- `optimizer.param_groups` to discover parameters (correctly handling
  multiple param groups).
- `optimizer.step()` to perform the actual update.

Because `p.grad` is the only thing being swapped, any optimizer that reads
`p.grad` during `step()` - SGD, AdamW, Adam, RMSprop, custom optimizers,
etc. - is transparently supported without special-casing.

## Usage

```python
import torch
from grad_transform import signed_square, step_with_transformed_gradients

model = torch.nn.Linear(4, 2)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

x, y = torch.randn(8, 4), torch.randn(8, 2)

optimizer.zero_grad(set_to_none=True)
loss = torch.nn.functional.mse_loss(model(x), y)
loss.backward()

# optimizer.step() runs against signed_square(grad); afterwards p.grad is
# restored to the original backward() values.
step_with_transformed_gradients(optimizer, signed_square)
```

## Running the tests

```bash
pip install pytest torch
pytest -q
```

Test coverage (`tests/test_grad_transform.py`):

- `signed_square` correctness against known input/output values.
- SGD reference-equivalence: wrapper output matches a hand-rolled
  "transform grad, then call native `SGD.step()`" reference (no reimplemented
  optimizer math).
- AdamW reference-equivalence: same, plus checks `exp_avg` / `exp_avg_sq`
  optimizer state matches, confirming AdamW's internal state is built from
  the transformed gradient.
- Gradient restoration: `p.grad` after the wrapper call equals the original
  `backward()` gradient (`torch.equal`).
- `p.grad is None` parameters are skipped, stay `None`, and no gradient is
  fabricated.
- Exception safety: if `optimizer.step()` raises, gradients are still fully
  restored.
- Multiple parameter groups are all transformed and restored correctly.

## Mini example / benchmark

```bash
python example.py    # tiny CPU MLP, synthetic data, SGD and AdamW
python benchmark.py  # baseline optimizer.step() vs. wrapper overhead
```

## Time / memory complexity

Let `N` = total number of elements across all gradient tensors.

- Transformation (`transform_fn`): `O(N)` time, and for `signed_square`
  specifically `O(N)` extra memory (it allocates a new tensor: `g * g.abs()`).
- Backup (clone) and restore: `O(N)` time and `O(N)` extra memory for the
  cloned originals.
- Overall: `O(N)` time and `O(N)` extra memory beyond the model's normal
  gradient buffers.

Because the wrapper keeps both the cloned original gradient *and* the
transformed gradient alive at the same time (plus whatever the optimizer
itself allocates, e.g. AdamW's `exp_avg`/`exp_avg_sq` state), peak memory
during a step can exceed one full copy of the gradients - roughly two extra
gradient-sized buffers (original + transformed) on top of the optimizer's
own state.

## Conceptual scope

`T(g) = g * abs(g)` is a **toy, deterministic gradient transformation**,
used here to exercise a gradient-interception / optimizer-integration
mechanism in a way that's easy to test and verify. It does *not* detect or
correct for gradient staleness, and it is not a staleness-mitigation
algorithm. A future research direction could generalize the callback to
something like `transform_fn(g, staleness_metadata)`, but no
staleness-aware logic is implemented here.

## Limitations / explicitly out of scope

- No asynchronous / RLVR-style staleness correction.
- No distributed training, FSDP, or ZeRO support.
- No sparse gradient support.
- No AMP / `GradScaler` integration.
- No GPU benchmarking (CPU-only benchmark included).

These are intentionally left out to keep the mechanism focused on
correctness and testability of gradient interception itself.

## Experiments

`experiments/` holds standalone research/benchmark scripts. None of them are
imported by, or modify, `grad_transform.py` - the production
`step_with_transformed_gradients` mechanism above is untouched by all of
them. Each is run directly, with no CLI arguments:

```bash
python experiments/reversible_tradeoff.py
python experiments/scale_study.py
python experiments/direction_consistency_study.py
```

- **`reversible_tradeoff.py`** - Compares the default safe (clone-based)
  gradient restoration against a reversible, in-place alternative that
  exploits `T(g) = sign(g) * g^2` being invertible
  (`T^-1(x) = sign(x) * sqrt(abs(x))`). Breaks down timing per component
  (clone/transform/step/restore vs. forward_transform/step/inverse_restore),
  measures both total and per-parameter peak memory, and rigorously tests
  round-trip precision (exhaustive over all float16/bfloat16 bit patterns;
  sampled + edge-case testing for float32).
- **`scale_study.py`** - Extends the safe-vs-reversible comparison across
  gradient sizes from ~0.2 MB to hundreds of MB, to check whether the
  timing/memory tradeoffs seen at small scale still hold once operations
  become memory-bandwidth-bound rather than dispatch-overhead-bound.
- **`direction_consistency_study.py`** - Observation-only research script
  asking whether a stale gradient's *directional consistency* (cosine
  similarity to a current/reference gradient) predicts its usefulness
  better than its *age* does. Reports measurements only; it does not
  implement any gradient reweighting or staleness-mitigation algorithm.

## Known risk areas (aliasing, restore, optimizer assumptions)

- **Tensor aliasing**: the original gradient is `clone()`d *before* any
  transform is applied or assigned back to `p.grad`, so an in-place-style
  transform (or an optimizer that mutates `p.grad` in place) cannot corrupt
  the saved original. As an extra guard, if `transform_fn` returns the exact
  same tensor object it was given (e.g. a no-op transform), the wrapper
  clones it again before assigning to `p.grad`, so the saved original and
  the live gradient are never the same object.
- **Restore is `try`/`finally`**: gradients are restored even if
  `optimizer.step()` raises.
- **AdamW / optimizer state**: the wrapper does not touch optimizer state
  (`exp_avg`, `exp_avg_sq`, momentum buffers, etc.) directly - it only
  swaps `p.grad` before calling `step()`. The AdamW reference-equivalence
  test confirms optimizer state ends up consistent with what native AdamW
  would produce given the transformed gradient.
- **Optimizer-specific assumptions**: none. The wrapper only reads
  `optimizer.param_groups` and calls `optimizer.step()`, so it makes no
  assumptions about a specific optimizer's update rule.
