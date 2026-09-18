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

Because `p.grad` is the only thing being swapped, the wrapper does not need
optimizer-specific update rules. This assumes `step()` directly consumes
the current `p.grad`, as in ordinary SGD and AdamW usage. Optimizers that
recompute gradients inside `step()` require additional integration.

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

For a small deterministic CPU validation of SGD and AdamW, run:

```bash
python validate.py
```

This uses fixed parameter tensors and manually assigned gradients, compares
each update with a native optimizer given `g * abs(g)` directly, and checks
exact restoration of the original gradient values after every step.

For the full test suite:

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
- A closure that recomputes gradients bypasses the transformation, while
  the pre-step gradient values are still restored.

## Mini example / benchmark

```bash
python example.py    # tiny CPU MLP, synthetic data, SGD and AdamW
python benchmark.py  # baseline optimizer.step() vs. wrapper overhead
```

## Time / memory complexity

Let `N` be the total number of elements in non-None gradients.

- For `signed_square`, cloning and transformation add `O(N)` time beyond
  the optimizer's own step, plus Python iteration over parameters.
  Restoration reassigns saved tensor references; it does not copy elements.
- Additional tensor space is `O(N)`. This clone-based restoration strategy
  requires at least one full gradient copy to preserve the original values.
  Transformed gradients also occupy storage, and `g * g.abs()` creates an
  additional temporary tensor for `abs(g)` while each gradient is processed.
- Peak memory is not a fixed number of gradient-sized buffers: it depends
  on tensor lifetimes, external references, and allocator behavior. The
  optimizer's own state and temporary allocations are separate costs.

These bounds apply to the included transform; an arbitrary `transform_fn`
can have different costs. `benchmark.py` measures CPU timing overhead, which
depends on tensor sizes and the optimizer.

## Conceptual scope

`T(g) = g * abs(g)` is a **toy, deterministic gradient transformation**,
used here to exercise a gradient-interception / optimizer-integration
mechanism in a way that's easy to test and verify. It does *not* detect or
correct for gradient staleness, and it is not a staleness-mitigation
algorithm. A future research direction could generalize the callback to
something like `transform_fn(g, staleness_metadata)`, but no
staleness-aware logic is implemented here.

## Limitations / explicitly out of scope

- `optimizer.step()` must directly use the current `p.grad`. If it
  recomputes gradients internally through a closure, those new gradients
  bypass the transformation, so closure-based optimizers are not guaranteed
  to work without additional integration.
- Transforms must not mutate their input or return an alias that can mutate
  the saved backup. Restoration preserves gradient values, not the original
  gradient tensor's object identity.
- No asynchronous / RLVR-style staleness correction.
- No distributed training, FSDP, or ZeRO support.
- Sparse gradients are untested; there is no explicit sparse-layout
  handling or rejection, so sparse support is not guaranteed.
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

- **Tensor aliasing**: the transform receives the saved gradient clone,
  so it must not mutate that input. The included `signed_square` returns
  a new tensor. If a transform returns its input unchanged, the wrapper
  clones it before assigning to `p.grad`; this guard does not protect
  against input mutation or other shared-storage views.
- **Restore is `try`/`finally`**: gradients are restored even if
  `optimizer.step()` raises.
- **AdamW / optimizer state**: the wrapper does not touch optimizer state
  (`exp_avg`, `exp_avg_sq`, momentum buffers, etc.) directly - it only
  swaps `p.grad` before calling `step()`. The AdamW reference-equivalence
  test confirms optimizer state ends up consistent with what native AdamW
  would produce given the transformed gradient.
- **Optimizer assumptions**: no specific update rule is reimplemented,
  but `step()` must consume the current `p.grad` without recomputing it.
  Forwarding a closure does not ensure that closure-generated gradients
  receive the transformation.
