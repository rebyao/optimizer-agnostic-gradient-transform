# Gradient Transformation Report

**Approach.** `step_with_transformed_gradients` iterates through
`optimizer.param_groups`, skips `grad=None`, and clones each remaining
gradient. With `signed_square`, it temporarily replaces `p.grad` with
`T(g) = g * abs(g)`, calls the original optimizer's `step()`, and restores
saved gradient values in `finally`, including when the step raises.
Restoration preserves values, not tensor identity.

**Why it is optimizer-agnostic.** The wrapper changes the gradients seen by
`step()` without reimplementing optimizer-specific update rules. Momentum,
Adam moments, and weight decay remain the native optimizer's responsibility.
This does not imply support for every optimizer. Reference comparisons
verify SGD with momentum and AdamW with weight decay; the AdamW unit test
also compares `exp_avg` and `exp_avg_sq` against native optimizer state.

**Assumptions / limitations.** The optimizer must consume the current
`p.grad`. Closure-recomputed gradients can overwrite transformed values
without being intercepted, as a focused SGD test demonstrates. Tests cover
skipping `grad=None`, multiple groups with distinct learning rates, and
restoration after exceptions. Exceptions do not roll back parameter or
optimizer-state changes. Sparse gradients have no explicit handling or
rejection and are untested. Transforms must not mutate or expose the saved
clone through a mutable alias; `signed_square` returns a new tensor.
AMP and distributed integration are outside the validated scope.

**Time and memory overhead.** For `N` non-None dense gradient elements,
cloning and transformation add `O(N)` time beyond the native step, plus
iteration over parameter entries. Restoration reassigns references.
Additional tensor memory is `O(N)`: restoration requires a full gradient
copy, while transformed outputs and temporaries such as `abs(g)` require
further storage. Peak memory depends on tensor lifetimes and allocator
behavior; optimizer state is a separate cost. These bounds apply to the
included transform, not arbitrary callbacks.

**Validation.** `.venv/bin/python validate.py` passed for SGD and AdamW.
Fixed CPU float64 parameters and three manual gradient vectors cover
positive, negative, and zero entries. Every step exactly matches a native
optimizer given explicitly transformed gradients, and original gradients
are exactly restored. `.venv/bin/python -m pytest -q` passed all **9 tests**,
covering transform values, reference updates, AdamW moments, restoration,
`None` gradients, multiple groups, exceptions, and the closure limitation.

**Extra experiments / Takeaways / Future work.** Standalone experiments
explore reversible restoration, scaling, and stale-gradient direction
consistency. Reversible restoration removes the full backup but retains
temporaries; exhaustive float16/bfloat16 checks found lossy round trips.
In the tiny SGD staleness proxy, direction achieved higher best F1 than age
on the observed samples with disjoint probe batches and normalized steps.
These findings clarify memory, numerical, and directional behavior without
changing the core mechanism or establishing real asynchronous-training
benefits. Future work could test broader models and gradient distributions,
and investigate memory-efficient restoration with exactness guarantees.
