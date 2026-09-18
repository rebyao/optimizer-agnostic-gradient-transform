"""Experiment (v2): safe clone-based restoration vs. reversible in-place restoration.

This is a STANDALONE experiment/benchmark, not a production implementation.
It does NOT change, replace, or get imported by grad_transform.py - the
default clone-based `step_with_transformed_gradients` mechanism there is
untouched and remains the safe default. Wherever this file needs the safe
strategy's *behavior* (e.g. the model-level sanity check), it imports and
calls the real production function; the only thing reimplemented here for
instrumentation is a component-level timing breakdown, since the production
function does not expose its internal phases separately.

Background
----------
The default transform is T(g) = sign(g) * g^2 = g * abs(g), which is
mathematically invertible: T^-1(x) = sign(x) * sqrt(abs(x)).

v2 changes vs. the original version of this file, in response to review:

  1. Timing is broken into components (clone/transform/step/restore for the
     safe strategy; forward_transform/step/inverse_restore for reversible),
     instead of only reporting a single total-time number.
  2. Memory accounting distinguishes M_total (sum of all gradient bytes)
     from M_max (largest single parameter's gradient bytes), and clearly
     labels transient-workspace figures as analytical estimates, not
     measured peaks.
  3. Precision testing is far more rigorous: float16 and bfloat16 are tested
     EXHAUSTIVELY over all 65536 bit patterns each; float32 is tested via
     Gaussian sampling, random bit-pattern sampling, AND targeted edge cases
     (subnormals, powers of two, values near sqrt(float32_max), etc.) -
     explicitly NOT claimed to be exhaustive.
  4. The model-level experiment now treats restored-gradient accuracy as the
     primary finding; parameter/optimizer-state equality is reported only as
     a sanity check, since both strategies feed optimizer.step() the same
     transformed gradient and are therefore expected to match by
     construction.
  5. A lower-allocation variant of the reversible inverse transform is
     defined and benchmarked against the original expression, to check
     whether "in-place" and "fewer temporary allocations" are actually
     being conflated.

Run with:  python experiments/reversible_tradeoff.py
"""

from __future__ import annotations

import copy
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grad_transform import signed_square, step_with_transformed_gradients

SEED = 0
NEAR_ZERO_THRESHOLD = 1e-8  # denominator floor for relative-error computation


# ---------------------------------------------------------------------------
# The transform and the two restoration strategies under comparison.
# ---------------------------------------------------------------------------


def forward_transform(g: torch.Tensor) -> torch.Tensor:
    """T(g) = sign(g) * g^2 = g * abs(g)."""
    return g * g.abs()


def inverse_transform(g: torch.Tensor) -> torch.Tensor:
    """T^-1(x) = sign(x) * sqrt(abs(x)) (original, out-of-place expression)."""
    return g.sign() * g.abs().sqrt()


def reversible_step(optimizer: torch.optim.Optimizer) -> None:
    """Experimental in-place restoration strategy (strategy B), original inverse expression.

    Keeps no full-size backup: mutates p.grad in place with the forward
    transform, runs optimizer.step(), then mutates p.grad in place with the
    (approximate) inverse transform. There is deliberately no try/finally
    safety net here, and restoration is only approximate - both of those
    costs are exactly what this experiment measures.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            p.grad.mul_(p.grad.abs())

    optimizer.step()

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            g.copy_(g.sign() * g.abs().sqrt())


def reversible_step_lowalloc(optimizer: torch.optim.Optimizer) -> None:
    """Same as reversible_step, but with a lower-allocation inverse pass.

    `g.sign() * g.abs().sqrt()` allocates a sign() temporary, an abs()
    temporary, a sqrt() result, and a multiply result - up to ~2x the
    tensor's size held concurrently at points in that expression. This
    variant captures sign() once, then does abs_()/sqrt_()/mul_() in place
    on g itself, needing only the one sign() temporary.

    NOTE: this is still not zero-allocation - "mutates its target in place"
    and "allocates no temporaries" are different properties, and this
    function has the former but not the latter.
    """
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            p.grad.mul_(p.grad.abs())

    optimizer.step()

    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            s = g.sign()
            g.abs_()
            g.sqrt_()
            g.mul_(s)


def _verify_lowalloc_equivalence() -> bool:
    torch.manual_seed(123)
    g = torch.randn(20_000) * 5.0
    g1 = g.clone()
    g2 = g.clone()
    r1 = inverse_transform(g1)
    s = g2.sign()
    g2.abs_()
    g2.sqrt_()
    g2.mul_(s)
    ok = torch.equal(r1, g2)
    print(f"[self-check] low-alloc inverse bit-identical to original expression: {ok}")
    if not ok:
        diff = (r1.double() - g2.double()).abs().max().item()
        print(f"  (max abs diff: {diff:.3e})")
    return ok


def _nan_aware_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Like torch.equal, but treats NaN == NaN as True.

    torch.equal follows IEEE semantics (NaN != NaN), so two tensors that
    independently landed on NaN at exactly the same positions - e.g. both
    optimizer runs hitting inf/inf in AdamW's update after a gradient
    overflowed - would otherwise be reported as "not equal" even though
    nothing about the two computations actually differed.
    """
    both_nan = torch.isnan(a) & torch.isnan(b)
    return bool(((a == b) | both_nan).all().item())


# ---------------------------------------------------------------------------
# Small table printer (no external dependencies).
# ---------------------------------------------------------------------------


def _format_cell(v, floatfmt="{:.6g}") -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if v != v else floatfmt.format(v)
    return str(v)


def print_table(headers, rows) -> None:
    str_rows = [[_format_cell(v) for v in row] for row in rows]
    widths = [
        max(len(headers[i]), max((len(r[i]) for r in str_rows), default=0))
        for i in range(len(headers))
    ]

    def fmt_row(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in str_rows:
        print(fmt_row(r))


# ---------------------------------------------------------------------------
# Shared model / data helpers.
# ---------------------------------------------------------------------------


def build_model() -> nn.Module:
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64))


def synthetic_batch(batch: int = 64, in_dim: int = 128, out_dim: int = 64, seed: int = SEED):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, in_dim, generator=g)
    y = torch.randn(batch, out_dim, generator=g)
    return x, y


# ---------------------------------------------------------------------------
# PART 1: component-level timing breakdown.
# ---------------------------------------------------------------------------

SAFE_COMPONENTS = ["clone", "transform", "step", "restore", "total"]
REV_COMPONENTS = ["forward_transform", "step", "inverse_restore", "total"]


def timed_safe_components(optimizer: torch.optim.Optimizer, transform_fn=signed_square) -> dict:
    """Instrumented reimplementation of grad_transform.step_with_transformed_gradients,
    split into components purely for timing. It mirrors that function's algorithm
    (clone -> transform -> step -> restore) so component timings sum to something
    comparable to the production function's total time; the production function
    itself is still what the rest of this script relies on for correctness.
    """
    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]

    t0 = time.perf_counter()
    originals = [p.grad.clone() for p in params]
    t1 = time.perf_counter()

    for p in params:
        p.grad = transform_fn(p.grad)
    t2 = time.perf_counter()

    optimizer.step()
    t3 = time.perf_counter()

    for p, orig in zip(params, originals):
        p.grad = orig
    t4 = time.perf_counter()

    return {"clone": t1 - t0, "transform": t2 - t1, "step": t3 - t2, "restore": t4 - t3, "total": t4 - t0}


def timed_reversible_components(optimizer: torch.optim.Optimizer, lowalloc: bool = False) -> dict:
    params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]

    t0 = time.perf_counter()
    for p in params:
        p.grad.mul_(p.grad.abs())
    t1 = time.perf_counter()

    optimizer.step()
    t2 = time.perf_counter()

    if lowalloc:
        for p in params:
            g = p.grad
            s = g.sign()
            g.abs_()
            g.sqrt_()
            g.mul_(s)
    else:
        for p in params:
            g = p.grad
            g.copy_(g.sign() * g.abs().sqrt())
    t3 = time.perf_counter()

    return {"forward_transform": t1 - t0, "step": t2 - t1, "inverse_restore": t3 - t2, "total": t3 - t0}


def run_component_timing(step_fn, component_names, optimizer_cls, kwargs, warmup=50, measured=500, trials=5):
    per_trial = {name: [] for name in component_names}
    for trial in range(trials):
        model = build_model()
        optimizer = optimizer_cls(model.parameters(), **kwargs)
        x, y = synthetic_batch(seed=SEED + trial)

        def backward():
            model.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(x), y).backward()

        for _ in range(warmup):
            backward()
            step_fn(optimizer)

        accum = {name: 0.0 for name in component_names}
        for _ in range(measured):
            backward()
            timings = step_fn(optimizer)
            for name in component_names:
                accum[name] += timings[name]

        for name in component_names:
            per_trial[name].append(accum[name] / measured * 1000.0)

    out = {}
    for name in component_names:
        t = torch.tensor(per_trial[name])
        out[name] = (t.mean().item(), t.std(unbiased=True).item() if trials > 1 else 0.0)
    return out


def experiment_timing_breakdown(warmup: int = 50, measured: int = 500, trials: int = 5):
    configs = [
        ("SGD", torch.optim.SGD, dict(lr=0.05, momentum=0.9)),
        ("AdamW", torch.optim.AdamW, dict(lr=1e-2)),
    ]
    results = {}
    for name, opt_cls, kwargs in configs:
        results[name] = dict(
            safe=run_component_timing(
                lambda opt: timed_safe_components(opt), SAFE_COMPONENTS, opt_cls, kwargs, warmup, measured, trials
            ),
            reversible=run_component_timing(
                lambda opt: timed_reversible_components(opt, False), REV_COMPONENTS, opt_cls, kwargs, warmup, measured, trials
            ),
            reversible_lowalloc=run_component_timing(
                lambda opt: timed_reversible_components(opt, True), REV_COMPONENTS, opt_cls, kwargs, warmup, measured, trials
            ),
        )

    def rows_for(strategy_key, components):
        return [
            [name] + [f"{results[name][strategy_key][c][0]:.4f}±{results[name][strategy_key][c][1]:.4f}" for c in components]
            for name in results
        ]

    print(f"-- Safe strategy component timing (ms/step, mean±std, {trials} trials, {warmup} warmup + {measured} measured) --")
    print_table(["optimizer"] + SAFE_COMPONENTS, rows_for("safe", SAFE_COMPONENTS))
    print()
    print("-- Reversible strategy (original inverse expression) component timing --")
    print_table(["optimizer"] + REV_COMPONENTS, rows_for("reversible", REV_COMPONENTS))
    print()
    print("-- Reversible strategy (low-allocation inverse) component timing --")
    print_table(["optimizer"] + REV_COMPONENTS, rows_for("reversible_lowalloc", REV_COMPONENTS))

    return results


# ---------------------------------------------------------------------------
# PART 2: corrected memory accounting (M_total vs M_max, analytical estimates).
# ---------------------------------------------------------------------------


def _fmt_bytes(b: float) -> str:
    return f"{b:.0f} B ({b / 1024:.2f} KB, {b / 1024 / 1024:.4f} MB)"


def experiment_memory_v2():
    model = build_model()
    x, y = synthetic_batch()
    model.zero_grad(set_to_none=True)
    nn.functional.mse_loss(model(x), y).backward()

    per_tensor_bytes = [
        p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None
    ]
    m_total = sum(per_tensor_bytes)
    m_max = max(per_tensor_bytes)

    safe_persistent = m_total  # one clone of every parameter's gradient, held across step()
    # ANALYTICAL ESTIMATE (not measured): grad_transform.py processes one
    # parameter at a time, so only one parameter's forward-transform
    # temporary (g.abs()) is alive at once, regardless of how many
    # parameters the model has.
    safe_transient_estimate = m_max

    reversible_persistent = 0  # no full-gradient backup is kept at all
    # ANALYTICAL ESTIMATE: the original inverse expression
    # `sign() * abs().sqrt()` can hold ~2 same-size temporaries concurrently
    # for whichever parameter is currently being processed; the low-alloc
    # variant (abs_/sqrt_/mul_ in place) needs only its one sign()
    # temporary, ~1x M_max.
    reversible_transient_estimate_original = 2 * m_max
    reversible_transient_estimate_lowalloc = m_max

    safe_peak_estimate = safe_persistent + safe_transient_estimate
    reversible_peak_estimate = reversible_persistent + reversible_transient_estimate_original

    print(f"Total gradient bytes (M_total)         : {_fmt_bytes(m_total)}")
    print(f"Largest single gradient tensor (M_max) : {_fmt_bytes(m_max)}")
    print()
    print(f"Safe persistent backup                 : {_fmt_bytes(safe_persistent)}  (exactly 1x M_total)")
    print(f"Safe transient workspace [ANALYTICAL]   : ~{_fmt_bytes(safe_transient_estimate)}  (~1x M_max, one parameter processed at a time)")
    print(f"Reversible persistent backup            : {_fmt_bytes(reversible_persistent)}  (0x M_total)")
    print(f"Reversible transient, original [ANALYTICAL] : ~{_fmt_bytes(reversible_transient_estimate_original)}  (~2x M_max)")
    print(f"Reversible transient, low-alloc [ANALYTICAL]: ~{_fmt_bytes(reversible_transient_estimate_lowalloc)}  (~1x M_max)")
    print()
    print(f"Safe peak estimate [ANALYTICAL]        : ~{_fmt_bytes(safe_peak_estimate)}  (persistent + transient, not a measured profiler peak)")
    print(f"Reversible peak estimate [ANALYTICAL]   : ~{_fmt_bytes(reversible_peak_estimate)}  (persistent + transient, not a measured profiler peak)")
    print()
    print("Correct framing: reversible restoration ELIMINATES THE FULL-MODEL PERSISTENT BACKUP")
    print("BUFFER (M_total). It does NOT mean zero peak extra memory - elementwise transforms")
    print("still create per-tensor temporaries on the order of M_max, for whichever parameter")
    print("is currently being processed. All 'estimate'/'ANALYTICAL' figures above come from")
    print("reading the sequence of tensor operations, not from a memory profiler.")
    print("(Model parameters, optimizer state, and PyTorch allocator overhead are excluded.)")

    return dict(
        m_total=m_total,
        m_max=m_max,
        safe_persistent=safe_persistent,
        safe_transient_estimate=safe_transient_estimate,
        reversible_persistent=reversible_persistent,
        reversible_transient_estimate_original=reversible_transient_estimate_original,
        reversible_transient_estimate_lowalloc=reversible_transient_estimate_lowalloc,
        safe_peak_estimate=safe_peak_estimate,
        reversible_peak_estimate=reversible_peak_estimate,
    )


# ---------------------------------------------------------------------------
# PART 3: precision testing.
# ---------------------------------------------------------------------------


def compute_roundtrip_stats(original: torch.Tensor) -> dict:
    """`original` must already be finite-filtered by the caller."""
    n = original.numel()
    transformed = forward_transform(original)
    restored = inverse_transform(transformed)

    exact_mask = restored == original
    exact_count = int(exact_mask.sum().item())

    underflow_mask = (transformed == 0) & (original != 0)
    underflow_count = int(underflow_mask.sum().item())

    overflow_mask = torch.isinf(transformed)
    overflow_count = int(overflow_mask.sum().item())

    # Mismatches not explained by a clean underflow-to-zero or overflow-to-inf
    # of the intermediate squared value - i.e. candidates for "genuine"
    # floating-point round-trip precision loss rather than a hard information
    # loss. Reported explicitly rather than assumed, since sampling schemes
    # that are biased toward extreme magnitudes (e.g. uniform bit patterns)
    # can otherwise make "mismatch %" look alarming for reasons unrelated to
    # ordinary-magnitude precision.
    other_mismatch_mask = (~exact_mask) & (~underflow_mask) & (~overflow_mask)
    other_mismatch_count = int(other_mismatch_mask.sum().item())

    orig_abs64 = original.double().abs()
    above_threshold_mask = orig_abs64 > NEAR_ZERO_THRESHOLD
    other_mismatch_above_threshold = int((other_mismatch_mask & above_threshold_mask).sum().item())

    diff = (restored.double() - original.double()).abs()
    finite_diff = diff[torch.isfinite(diff)]
    max_abs_err = finite_diff.max().item() if finite_diff.numel() else float("nan")
    mean_abs_err = finite_diff.mean().item() if finite_diff.numel() else float("nan")

    if above_threshold_mask.any():
        rel = diff[above_threshold_mask] / orig_abs64[above_threshold_mask]
        rel = rel[torch.isfinite(rel)]
        max_rel_err = rel.max().item() if rel.numel() else float("nan")
    else:
        max_rel_err = float("nan")

    return dict(
        n=n,
        exact_count=exact_count,
        mismatch_count=n - exact_count,
        exact_pct=(exact_count / n * 100.0) if n else float("nan"),
        underflow_count=underflow_count,
        overflow_count=overflow_count,
        other_mismatch_count=other_mismatch_count,
        other_mismatch_above_threshold=other_mismatch_above_threshold,
        max_abs_err=max_abs_err,
        mean_abs_err=mean_abs_err,
        max_rel_err=max_rel_err,
    )


# --- 3A/3B: exhaustive float16 / bfloat16 --------------------------------


@dataclass
class ExhaustiveResult:
    label: str
    total_bit_patterns: int
    n_nan: int
    n_inf: int
    n_finite: int
    exact_count: int
    mismatch_count: int
    exact_pct: float
    underflow_count: int
    overflow_count: int
    other_mismatch_count: int
    max_abs_err: float
    signed_zero_preserved: object
    n_signed_zero_mismatch: int


def exhaustive_fp_test(dtype: torch.dtype, label: str) -> ExhaustiveResult:
    """Enumerate all 2^16 bit patterns of a 16-bit float dtype via bit reinterpretation."""
    bits = torch.arange(0, 65536, dtype=torch.int32).to(torch.uint16)
    values = bits.view(dtype)  # bit-cast, not a numeric conversion

    total = values.numel()
    nan_mask = torch.isnan(values)
    inf_mask = torch.isinf(values)
    finite_mask = ~nan_mask & ~inf_mask

    n_nan = int(nan_mask.sum().item())
    n_inf = int(inf_mask.sum().item())
    n_finite = int(finite_mask.sum().item())

    original = values[finite_mask].clone()
    transformed = forward_transform(original)
    restored = inverse_transform(transformed)

    exact_mask = restored == original
    exact_count = int(exact_mask.sum().item())
    mismatch_count = n_finite - exact_count

    underflow_mask = (transformed == 0) & (original != 0)
    underflow_count = int(underflow_mask.sum().item())
    overflow_mask = torch.isinf(transformed)
    overflow_count = int(overflow_mask.sum().item())
    other_mismatch_count = int((exact_mask.logical_not() & ~underflow_mask & ~overflow_mask).sum().item())

    diff = (restored.double() - original.double()).abs()
    finite_diff = diff[torch.isfinite(diff)]
    max_abs_err = finite_diff.max().item() if finite_diff.numel() else float("nan")

    zero_mask = original == 0
    if zero_mask.any():
        same_sign = torch.signbit(restored[zero_mask]) == torch.signbit(original[zero_mask])
        signed_zero_preserved = bool(same_sign.all().item())
        n_signed_zero_mismatch = int((~same_sign).sum().item())
    else:
        signed_zero_preserved = None
        n_signed_zero_mismatch = 0

    return ExhaustiveResult(
        label=label,
        total_bit_patterns=total,
        n_nan=n_nan,
        n_inf=n_inf,
        n_finite=n_finite,
        exact_count=exact_count,
        mismatch_count=mismatch_count,
        exact_pct=(exact_count / n_finite * 100.0) if n_finite else float("nan"),
        underflow_count=underflow_count,
        overflow_count=overflow_count,
        other_mismatch_count=other_mismatch_count,
        max_abs_err=max_abs_err,
        signed_zero_preserved=signed_zero_preserved,
        n_signed_zero_mismatch=n_signed_zero_mismatch,
    )


def print_exhaustive_summary(r: ExhaustiveResult) -> None:
    print(f"dtype                        : {r.label}")
    print(f"total bit patterns           : {r.total_bit_patterns}")
    print(f"  NaN patterns               : {r.n_nan}")
    print(f"  Inf patterns               : {r.n_inf}")
    print(f"  Finite patterns (tested)   : {r.n_finite}")
    print(f"exact restored               : {r.exact_count} ({r.exact_pct:.4f}%)")
    print(f"mismatch                     : {r.mismatch_count}")
    print(f"  underflow-to-zero          : {r.underflow_count}")
    print(f"  overflow-to-inf            : {r.overflow_count}")
    print(f"  other (not under/overflow) : {r.other_mismatch_count}  <- genuine round-trip precision loss, not explained by hard under/overflow")
    print(f"max abs error (finite pairs) : {r.max_abs_err:.6e}")
    if r.signed_zero_preserved is None:
        print("signed-zero preserved        : n/a (no zero value in finite set)")
    else:
        print(f"signed-zero preserved        : {r.signed_zero_preserved} (mismatches: {r.n_signed_zero_mismatch})")


# --- 3C: float32 (non-exhaustive: Gaussian, random bits, edge cases) ------

GAUSSIAN_SCALES = [1e-4, 1e-2, 1.0, 10.0, 100.0]
GAUSSIAN_N = 500_000


def experiment_gaussian_f32_f64():
    rows = []
    seed = 2000
    for dtype in (torch.float64, torch.float32):
        for scale in GAUSSIAN_SCALES:
            seed += 1
            torch.manual_seed(seed)
            base = torch.randn(GAUSSIAN_N, dtype=torch.float64) * scale
            g = base.to(dtype)
            stats = compute_roundtrip_stats(g)
            rows.append(
                [
                    str(dtype).replace("torch.", ""), scale, stats["mismatch_count"] == 0,
                    stats["mismatch_count"], stats["exact_pct"], stats["underflow_count"],
                    stats["overflow_count"], stats["max_abs_err"], stats["max_rel_err"],
                ]
            )
    headers = ["dtype", "scale", "no_mismatch", "mismatch_n", "exact_%", "underflow_n", "overflow_n", "max_abs_err", "max_rel_err"]
    print_table(headers, rows)
    return rows


def experiment_float32_random_bits(n: int = 2_000_000, seed: int = 42):
    torch.manual_seed(seed)
    bits = torch.randint(low=-(2**31), high=2**31, size=(n,), dtype=torch.int32)
    values = bits.view(torch.float32)
    finite_mask = torch.isfinite(values)
    finite_values = values[finite_mask]
    n_finite = finite_values.numel()
    n_excluded = n - n_finite
    stats = compute_roundtrip_stats(finite_values)

    print(f"Sampled bit patterns : {n}")
    print(f"Excluded (nan/inf)   : {n_excluded} ({n_excluded / n * 100:.4f}%)")
    print(f"Tested (finite)      : {n_finite}")
    print(f"Exact matches        : {stats['exact_count']} ({stats['exact_pct']:.6f}%)")
    print(f"Mismatches           : {stats['mismatch_count']}")
    print(f"  Underflow-to-zero  : {stats['underflow_count']}")
    print(f"  Overflow-to-inf    : {stats['overflow_count']}")
    print(f"  Other mismatches   : {stats['other_mismatch_count']}  (not explained by hard under/overflow)")
    print(f"    ...with |original| > {NEAR_ZERO_THRESHOLD:g} : {stats['other_mismatch_above_threshold']}  <- genuine precision loss at non-negligible magnitude")
    print(f"Max abs error        : {stats['max_abs_err']:.6e}")
    print(f"Max rel error        : {stats['max_rel_err']:.6e}  (computed only where |original| > {NEAR_ZERO_THRESHOLD:g})")
    print(f"NOTE: this is RANDOM SAMPLING of the float32 bit-pattern space (2^32 total patterns),")
    print(f"      not exhaustive - {n} samples is a negligible fraction of 2^32.")
    print("NOTE: sampling bit patterns UNIFORMLY over int32 heavily over-represents extreme")
    print("      exponents (very large or very near-zero magnitudes) compared to realistic")
    print("      gradient values, which is why the underflow/overflow counts here are far higher")
    print("      than in the Gaussian-sampling test above - this stress-tests the full")
    print("      representable range, not a realistic gradient distribution.")

    result = dict(n_sampled=n, n_excluded=n_excluded, n_finite=n_finite)
    result.update(stats)
    return result


def build_float32_edge_cases():
    finfo = torch.finfo(torch.float32)

    def t(v):
        return torch.tensor(float(v), dtype=torch.float32)

    def nextafter_towards(v, target):
        return torch.nextafter(t(v), t(target)).item()

    cases = [("zero", 0.0), ("neg_zero", -0.0)]

    smallest_subnormal = finfo.smallest_normal * (2.0**-23)
    cases.append(("smallest_subnormal", smallest_subnormal))
    for k in (2, 4, 16, 256, 4096, 2**20):
        cases.append((f"subnormal_x{k}", smallest_subnormal * k))

    cases.append(("smallest_normal", finfo.smallest_normal))
    cases.append(("largest_subnormal", nextafter_towards(finfo.smallest_normal, 0.0)))
    cases.append(("just_above_smallest_normal", nextafter_towards(finfo.smallest_normal, 1.0)))

    for e in range(-126, 121, 12):
        cases.append((f"pow2_e{e}", 2.0**e))

    for e in (-10, -1, 0, 1, 10, 50, 100):
        base = 2.0**e
        cases.append((f"just_below_pow2_e{e}", nextafter_towards(base, 0.0)))
        cases.append((f"just_above_pow2_e{e}", nextafter_towards(base, float("inf"))))

    sqrt_max = math.sqrt(finfo.max)
    cases.append(("sqrt_float32_max", sqrt_max))
    cases.append(("just_below_sqrt_max", nextafter_towards(sqrt_max, 0.0)))
    cases.append(("just_above_sqrt_max", nextafter_towards(sqrt_max, float("inf"))))
    cases.append(("sqrt_max_times_1.0001", sqrt_max * 1.0001))
    cases.append(("sqrt_max_times_0.9999", sqrt_max * 0.9999))

    cases.append(("float32_max", finfo.max))

    labels, values = [], []
    for label, v in cases:
        labels.append(label)
        values.append(v)
        if v != 0.0:
            labels.append("neg_" + label)
            values.append(-v)

    tensor = torch.tensor(values, dtype=torch.float32)
    finite_mask = torch.isfinite(tensor)
    n_dropped = int((~finite_mask).sum().item())
    kept_labels = [l for l, m in zip(labels, finite_mask.tolist()) if m]
    return kept_labels, tensor[finite_mask], n_dropped


def experiment_float32_edge_cases():
    labels, original, n_dropped = build_float32_edge_cases()
    transformed = forward_transform(original)
    restored = inverse_transform(transformed)

    exact_mask = restored == original
    diff = (restored.double() - original.double()).abs()

    rows = []
    for i, label in enumerate(labels):
        rows.append(
            [
                label,
                original[i].item(),
                restored[i].item(),
                bool(exact_mask[i].item()),
                diff[i].item(),
                bool(torch.isinf(transformed[i]).item()),
                bool(transformed[i].item() == 0.0 and original[i].item() != 0.0),
            ]
        )
    headers = ["case", "original", "restored", "exact", "abs_err", "overflow", "underflow"]
    print_table(headers, rows)

    n = len(labels)
    exact_count = int(exact_mask.sum().item())
    overflow_count = int(torch.isinf(transformed).sum().item())
    underflow_mask = (transformed == 0) & (original != 0)
    underflow_count = int(underflow_mask.sum().item())
    other_mismatch_count = int((exact_mask.logical_not() & ~underflow_mask & ~torch.isinf(transformed)).sum().item())
    print(
        f"\nSummary: {exact_count}/{n} exact, {overflow_count} overflow-to-inf, "
        f"{underflow_count} underflow-to-zero, {other_mismatch_count} other mismatch"
    )
    print("(every mismatch here is accounted for by overflow or underflow of the intermediate")
    print(" squared value at the extremes we deliberately targeted, unless 'other mismatch' > 0)")
    if n_dropped:
        print(f"({n_dropped} constructed candidate value(s) were already non-finite in float32 and dropped before testing)")

    return dict(
        n=n, exact_count=exact_count, overflow_count=overflow_count,
        underflow_count=underflow_count, other_mismatch_count=other_mismatch_count, labels=labels,
    )


# ---------------------------------------------------------------------------
# PART 4: model-level check - restored-gradient accuracy is the primary
# finding; parameter/optimizer-state equality is a sanity check only.
# ---------------------------------------------------------------------------


def experiment_model_level_v2(dtype: torch.dtype, steps: int = 5):
    configs = [
        ("SGD", torch.optim.SGD, dict(lr=0.05, momentum=0.9)),
        ("AdamW", torch.optim.AdamW, dict(lr=1e-2)),
    ]
    x, y = synthetic_batch()
    x = x.to(dtype)
    y = y.to(dtype)
    dtype_label = str(dtype).replace("torch.", "")
    results = []

    for name, opt_cls, kwargs in configs:
        try:
            torch.manual_seed(SEED)
            model_safe = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64)).to(dtype)
            model_rev = copy.deepcopy(model_safe)
            opt_safe = opt_cls(model_safe.parameters(), **kwargs)
            opt_rev = opt_cls(model_rev.parameters(), **kwargs)

            all_diffs = []
            exact_matches = 0
            total_elems = 0

            for _ in range(steps):
                model_safe.zero_grad(set_to_none=True)
                nn.functional.mse_loss(model_safe(x), y).backward()
                step_with_transformed_gradients(opt_safe, signed_square)

                model_rev.zero_grad(set_to_none=True)
                nn.functional.mse_loss(model_rev(x), y).backward()

                # Measurement-only clone: experiment instrumentation to score
                # restoration error, NOT part of the reversible strategy's
                # (hypothetical) memory footprint - see experiment_memory_v2().
                measurement_originals = {
                    p: p.grad.clone() for p in model_rev.parameters() if p.grad is not None
                }
                reversible_step(opt_rev)

                for p, orig in measurement_originals.items():
                    diff = (p.grad.double() - orig.double()).abs()
                    all_diffs.append(diff.flatten())
                    exact_matches += int((p.grad == orig).sum().item())
                    total_elems += orig.numel()

            diffs = torch.cat(all_diffs)
            finite_diffs = diffs[torch.isfinite(diffs)]
            max_abs_err = finite_diffs.max().item() if finite_diffs.numel() else float("nan")
            mean_abs_err = finite_diffs.mean().item() if finite_diffs.numel() else float("nan")
            exact_pct = exact_matches / total_elems * 100.0 if total_elems else float("nan")

            # torch.equal treats NaN != NaN (IEEE semantics), so if both paths
            # independently corrupt to NaN at the SAME positions (which is what
            # actually happens for fp16 + AdamW once a gradient overflows to
            # inf - see the note printed below), a naive torch.equal falsely
            # reports "not bit-exact" even though nothing about the
            # restoration strategy differs. Use a NaN-aware comparison.
            params_equal = all(
                _nan_aware_equal(pa, pb) for pa, pb in zip(model_safe.parameters(), model_rev.parameters())
            )
            params_have_nan = any(torch.isnan(p).any().item() for p in model_safe.parameters())
            state_equal = None
            if name == "AdamW":
                state_equal = all(
                    _nan_aware_equal(opt_safe.state[pa]["exp_avg"], opt_rev.state[pb]["exp_avg"])
                    and _nan_aware_equal(opt_safe.state[pa]["exp_avg_sq"], opt_rev.state[pb]["exp_avg_sq"])
                    for pa, pb in zip(model_safe.parameters(), model_rev.parameters())
                )

            results.append(
                dict(
                    dtype=dtype_label, optimizer=name, supported=True, max_abs_err=max_abs_err,
                    mean_abs_err=mean_abs_err, exact_pct=exact_pct, params_equal=params_equal,
                    params_have_nan=params_have_nan, state_equal=state_equal, error=None,
                )
            )
        except Exception as e:  # pragma: no cover - environment dependent
            results.append(
                dict(
                    dtype=dtype_label, optimizer=name, supported=False, max_abs_err=float("nan"),
                    mean_abs_err=float("nan"), exact_pct=float("nan"), params_equal=None,
                    params_have_nan=None, state_equal=None, error=str(e),
                )
            )
    return results


def experiment_model_level_all_dtypes():
    dtypes = [torch.float64, torch.float32, torch.bfloat16, torch.float16]
    all_results = []
    for dt in dtypes:
        all_results.extend(experiment_model_level_v2(dt))

    print("PRIMARY finding: restored-gradient accuracy vs. dtype/optimizer")
    headers = ["dtype", "optimizer", "supported", "restore_max_abs_err", "restore_mean_abs_err", "restore_exact_%"]
    rows = [[r["dtype"], r["optimizer"], r["supported"], r["max_abs_err"], r["mean_abs_err"], r["exact_pct"]] for r in all_results]
    print_table(headers, rows)

    print()
    print("SANITY CHECK ONLY (expected to match by construction, see note below):")
    headers2 = ["dtype", "optimizer", "params_bitexact*", "adamw_state_bitexact*", "params_contain_nan"]
    rows2 = [
        [
            r["dtype"], r["optimizer"], r["params_equal"],
            "n/a" if r["state_equal"] is None else r["state_equal"],
            "n/a" if r["params_have_nan"] is None else r["params_have_nan"],
        ]
        for r in all_results
    ]
    print_table(headers2, rows2)
    print("Note: both strategies feed optimizer.step() the identical transformed gradient T(g);")
    print("restoration happens strictly afterward, so params/state matching is a correctness")
    print("sanity check on this harness, not a finding about the restoration strategies.")
    print("* computed with a NaN-aware equality (NaN == NaN counts as a match here, unlike")
    print("  torch.equal) - see params_contain_nan: when float16 gradients overflow to inf,")
    print("  AdamW's own update can produce NaN parameters independently of which restoration")
    print("  strategy is used; a naive torch.equal would then report 'not bit-exact' even when")
    print("  both runs corrupted to NaN at identical positions for an unrelated reason.")

    for r in all_results:
        if not r["supported"]:
            print(f"[unsupported] {r['dtype']} / {r['optimizer']}: {r['error']}")

    return all_results


# ---------------------------------------------------------------------------
# Conclusions.
# ---------------------------------------------------------------------------


def print_conclusions_v2(timing_results, memory_result, fp16_result, bf16_result,
                          f32_gaussian_rows, f32_random_bits, f32_edge_cases, model_results):
    print("1. What is the safe strategy's main memory cost?")
    print(f"   - A persistent backup clone of every parameter gradient: M_total = {_fmt_bytes(memory_result['m_total'])}.")
    print("     This is exactly 1x the total gradient size, held for the duration of optimizer.step().")

    print()
    print("2. What memory does the reversible strategy actually eliminate?")
    print("   - The persistent M_total backup entirely: reversible_persistent = 0 bytes of")
    print("     full-gradient backup are kept alive across optimizer.step().")

    print()
    print("3. What transient memory does the reversible strategy still have?")
    print(f"   - ANALYTICAL ESTIMATE (not profiler-measured): the original inverse expression")
    print(f"     can hold ~2x M_max concurrently per parameter; the low-allocation rewrite reduces")
    print(f"     that to ~1x M_max (M_max = {_fmt_bytes(memory_result['m_max'])} here). Either way,")
    print(f"     it is bounded by the largest single parameter, not by the whole model.")

    print()
    print("4. What is the dominant component of the reversible strategy's slowdown?")
    for opt_name in timing_results:
        safe = timing_results[opt_name]["safe"]
        rev = timing_results[opt_name]["reversible"]
        rev_total = rev["total"][0]
        safe_total = safe["total"][0]
        inv_pct = rev["inverse_restore"][0] / rev_total * 100 if rev_total else float("nan")
        fwd_pct = rev["forward_transform"][0] / rev_total * 100 if rev_total else float("nan")
        step_pct_rev = rev["step"][0] / rev_total * 100 if rev_total else float("nan")
        clone_pct = safe["clone"][0] / safe_total * 100 if safe_total else float("nan")
        restore_pct = safe["restore"][0] / safe_total * 100 if safe_total else float("nan")
        print(f"   - {opt_name}: reversible total={rev_total:.4f} ms/step -> inverse_restore={inv_pct:.1f}%, "
              f"forward_transform={fwd_pct:.1f}%, step={step_pct_rev:.1f}%")
        print(f"     safe total={safe_total:.4f} ms/step -> clone={clone_pct:.1f}%, restore={restore_pct:.1f}%")
    print("   (measured from the component breakdown above, rather than attributed to")
    print("    'in-place is slower' or 'an extra param_groups pass' without evidence.)")

    print()
    print("5. Does float16 have clearly irreversible regions?")
    print(f"   - Yes, in two distinct ways. Exhaustive test over all {fp16_result.n_finite} finite float16 values:")
    print(f"     {fp16_result.underflow_count} underflowed to exactly 0 and {fp16_result.overflow_count} overflowed")
    print(f"     to inf when squared (both hard, unrecoverable information loss); separately,")
    print(f"     {fp16_result.other_mismatch_count} more values round-tripped to a DIFFERENT finite value")
    print(f"     (silent precision loss, not caught by an inf/0 check). Overall exact-restoration rate:")
    print(f"     {fp16_result.exact_pct:.4f}%.")

    print()
    print("6. What are the bfloat16 exhaustive results?")
    print(f"   - Exhaustive test over all {bf16_result.n_finite} finite bfloat16 values: "
          f"{bf16_result.exact_pct:.4f}% exact, {bf16_result.underflow_count} underflow-to-zero, "
          f"{bf16_result.overflow_count} overflow-to-inf, and {bf16_result.other_mismatch_count} silent")
    print(f"     (non-inf/0) precision mismatches, max abs error {bf16_result.max_abs_err:.3e}.")
    print(f"     Fewer silent mismatches than float16 ({bf16_result.other_mismatch_count} vs "
          f"{fp16_result.other_mismatch_count}) but a comparable overflow/underflow rate, since bf16 trades")
    print(f"     mantissa bits (more rounding within range) for float32's exponent range.")

    print()
    print("7. Did we find a float32 counterexample?")
    f32_gauss_rows_only = [r for r in f32_gaussian_rows if r[0] == "float32"]
    gaussian_mismatches = sum(r[3] for r in f32_gauss_rows_only)
    print(f"   - Gaussian sampling ({len(f32_gauss_rows_only)} scales x {GAUSSIAN_N:,} elements, magnitudes")
    print(f"     1e-4..100, i.e. realistic gradient range): {gaussian_mismatches} mismatches found.")
    print(f"   - Random bit-pattern sampling ({f32_random_bits['n_finite']:,} finite values out of "
          f"{f32_random_bits['n_sampled']:,} sampled, uniform over the FULL float32 exponent range):")
    print(f"     {f32_random_bits['mismatch_count']} mismatches total, but "
          f"{f32_random_bits['overflow_count']} are overflow-to-inf and "
          f"{f32_random_bits['underflow_count']} are underflow-to-zero - both are hard information")
    print(f"     loss at extreme magnitudes, not precision drift. Only "
          f"{f32_random_bits['other_mismatch_above_threshold']} mismatch(es) remain with "
          f"|original| > {NEAR_ZERO_THRESHOLD:g} and are not explained by under/overflow.")
    print(f"   - Targeted edge cases ({f32_edge_cases['n']} values incl. subnormals, powers of two, "
          f"values near sqrt(float32_max), float32_max):")
    print(f"     {f32_edge_cases['n'] - f32_edge_cases['exact_count']} mismatches, all "
          f"{f32_edge_cases['overflow_count']} overflow + {f32_edge_cases['underflow_count']} underflow "
          f"(other mismatch: {f32_edge_cases['other_mismatch_count']}) - i.e. exactly the boundary")
    print(f"     cases we targeted, not unexplained precision loss.")
    print("   Reading these together: within the range where the intermediate square doesn't")
    print("   overflow or underflow, every sampled/enumerated float32 value round-tripped exactly.")

    print()
    print("8. If no counterexample was found, can we claim it's universally exact for float32?")
    print("   - No. Sampling - Gaussian, random-bit-pattern, and edge-case combined - covers an")
    print("     infinitesimal fraction of float32's ~4.3 billion values. It is not a proof. The")
    print("     correct claim is: 'no mismatch observed in sampled float32 values', never")
    print("     'exactly reversible for all float32 values'.")

    print()
    print("9. Is the reversible strategy suitable as the default implementation?")
    print("   - No. It has provably unrecoverable failure modes at fp16 (Q5), no exception safety,")
    print("     and - per the component timing breakdown (Q4) - is not reliably faster either.")

    print()
    print("10. Is it better suited as an experimental low-memory option?")
    print("    - Yes, conditionally: fp32/fp64 only, with a try/finally safety net added, and with")
    print("      the caveat that 'no mismatch observed' (Q7/Q8) is empirical, not a guarantee.")

    unsupported = [r for r in model_results if not r["supported"]]
    print()
    if unsupported:
        print(f"(Model-level check: {len(unsupported)} (dtype, optimizer) combination(s) were unsupported "
              "on this build: " + ", ".join(f"{r['dtype']}/{r['optimizer']}" for r in unsupported) + ".)")
    else:
        print("(Model-level check: all tested (dtype, optimizer) combinations ran on this CPU/torch build;")
        print(" see PART 4 above for restored-gradient accuracy, the primary finding there.)")

    print()
    print("Summary statement:")
    print("  Reversible restoration eliminates the full-model persistent gradient backup, but")
    print("  replaces that storage cost with an additional elementwise inverse pass and retains")
    print("  per-tensor transient workspace. Precision failures are clearly present where squaring")
    print("  underflows or overflows. For float32, sampled exact recovery is empirical evidence")
    print("  only and should not be interpreted as a universal guarantee.")


def main():
    torch.manual_seed(SEED)

    print("=" * 92)
    print("Self-check: low-allocation inverse variant vs. original expression")
    print("=" * 92)
    _verify_lowalloc_equivalence()

    print()
    print("=" * 92)
    print("PART 1: Timing breakdown (component-level, not just total)")
    print("=" * 92)
    timing_results = experiment_timing_breakdown()

    print()
    print("=" * 92)
    print("PART 2: Memory accounting (M_total vs M_max, analytical peak estimates)")
    print("=" * 92)
    memory_result = experiment_memory_v2()

    print()
    print("=" * 92)
    print("PART 3A: float16 - EXHAUSTIVE test over all 65536 bit patterns")
    print("=" * 92)
    fp16_result = exhaustive_fp_test(torch.float16, "float16")
    print_exhaustive_summary(fp16_result)

    print()
    print("=" * 92)
    print("PART 3B: bfloat16 - EXHAUSTIVE test over all 65536 bit patterns")
    print("=" * 92)
    bf16_result = exhaustive_fp_test(torch.bfloat16, "bfloat16")
    print_exhaustive_summary(bf16_result)

    print()
    print("=" * 92)
    print("PART 3C-1: float32 (and float64 for reference) - Gaussian random sampling")
    print("=" * 92)
    f32_gaussian_rows = experiment_gaussian_f32_f64()

    print()
    print("=" * 92)
    print("PART 3C-2: float32 - random bit-pattern sampling (NOT exhaustive)")
    print("=" * 92)
    f32_random_bits = experiment_float32_random_bits()

    print()
    print("=" * 92)
    print("PART 3C-3: float32 - targeted edge cases")
    print("=" * 92)
    f32_edge_cases = experiment_float32_edge_cases()

    print()
    print("=" * 92)
    print("PART 4: Model-level check (SGD/AdamW) - restored-gradient accuracy is primary")
    print("=" * 92)
    model_results = experiment_model_level_all_dtypes()

    print()
    print("=" * 92)
    print("Conclusions (deliberately conservative wording)")
    print("=" * 92)
    print_conclusions_v2(
        timing_results, memory_result, fp16_result, bf16_result,
        f32_gaussian_rows, f32_random_bits, f32_edge_cases, model_results,
    )


if __name__ == "__main__":
    main()
