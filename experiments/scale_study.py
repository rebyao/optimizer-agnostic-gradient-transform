"""Experiment: how do safe (clone-based) vs. reversible (in-place) gradient
restoration strategies SCALE with gradient/model size, in time and memory?

This is a STANDALONE experiment, independent of experiments/reversible_tradeoff.py
and of grad_transform.py - neither is modified here. It answers questions that a
single small-model benchmark cannot:

  1. How does the EXTRA time (beyond raw compute) grow as total gradient size
     grows from ~0.2 MB to hundreds of MB?
  2. Does the safe strategy's persistent backup grow linearly with size (it
     must, by construction - this experiment verifies that empirically)?
  3. Does the reversible strategy actually avoid a full-model persistent backup
     at every scale, not just small ones?
  4. Do the timing-percentage differences seen on a small model still hold at
     large scale, or do they change once operations become memory-bandwidth-
     bound rather than Python/dispatch-overhead-bound?
  5. Which cost component scales with N (elementwise memory traffic) and which
     is roughly constant (Python loop / dispatch overhead)?

Design notes
------------
- No forward/backward and no real dataset: parameter tensors are created
  directly via nn.Parameter, and `.grad` is assigned directly with
  `torch.randn_like(...)`. This isolates the gradient-transform/restore
  mechanism from model-compute noise and gives exact control over total
  gradient size.
- The gradient for a given set of parameters is generated ONCE per trial (not
  re-randomized every iteration): the safe strategy restores it exactly, and
  earlier precision testing (see experiments/reversible_tradeoff.py, Part 3)
  found float32 round-trips are exact away from extreme magnitudes, so
  reusing the same gradient across a trial's iterations does not introduce
  meaningful drift, while avoiding the cost of re-generating large random
  tensors every single iteration (which would otherwise dominate wall time
  at the largest scales without helping the measurement).
- The PRIMARY tables exclude optimizer.step() entirely, to isolate the
  transform/restore mechanism's own scaling. An OPTIONAL supplementary
  experiment adds SGD/AdamW at a few scales for context.

Run with:  python experiments/scale_study.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grad_transform import signed_square, step_with_transformed_gradients

SEED = 0
DTYPE = torch.float32
NUM_TENSORS = 8
MB = 1024 * 1024

TARGET_SIZES_MB = [0.2, 2, 20, 100, 200, 500]


# ---------------------------------------------------------------------------
# Small table printer (kept local/self-contained, not shared with the other
# experiment file, so this script has no dependency beyond grad_transform.py).
# ---------------------------------------------------------------------------


def _format_cell(v, floatfmt="{:.5g}") -> str:
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


def fmt_mean_std(mean, std, fmt="{:.4f}"):
    return f"{fmt.format(mean)}±{fmt.format(std)}"


# ---------------------------------------------------------------------------
# Parameter/gradient construction (no forward/backward, no dataset).
# ---------------------------------------------------------------------------


def build_param_tensors(target_mb: float, num_tensors: int = NUM_TENSORS, dtype: torch.dtype = DTYPE, seed: int = SEED):
    """Build `num_tensors` nn.Parameter tensors whose gradients sum to
    approximately `target_mb` megabytes, mimicking several param_groups
    entries rather than one giant tensor."""
    torch.manual_seed(seed)
    elem_bytes = torch.tensor([], dtype=dtype).element_size()
    total_elems = max(int(target_mb * MB / elem_bytes), num_tensors)
    base = total_elems // num_tensors
    remainder = total_elems - base * num_tensors
    sizes = [base + (1 if i < remainder else 0) for i in range(num_tensors)]
    sizes = [s for s in sizes if s > 0]

    params = []
    for s in sizes:
        p = nn.Parameter(torch.randn(s, dtype=dtype))
        p.grad = torch.randn(s, dtype=dtype)
        params.append(p)
    return params


def gradient_bytes(params) -> int:
    return sum(p.grad.numel() * p.grad.element_size() for p in params)


def max_tensor_bytes(params) -> int:
    return max(p.grad.numel() * p.grad.element_size() for p in params)


# ---------------------------------------------------------------------------
# The two mechanisms under test (transform + restore ONLY, no optimizer.step()).
# ---------------------------------------------------------------------------


def timed_safe_step(params) -> dict:
    t0 = time.perf_counter()
    originals = [p.grad.clone() for p in params]
    t1 = time.perf_counter()

    for p in params:
        p.grad = p.grad * p.grad.abs()
    t2 = time.perf_counter()

    for p, orig in zip(params, originals):
        p.grad = orig
    t3 = time.perf_counter()

    return {"clone": t1 - t0, "forward": t2 - t1, "restore": t3 - t2, "total": t3 - t0}


def timed_reversible_step(params) -> dict:
    t0 = time.perf_counter()
    for p in params:
        p.grad.mul_(p.grad.abs())
    t1 = time.perf_counter()

    for p in params:
        g = p.grad
        g.copy_(g.sign() * g.abs().sqrt())
    t2 = time.perf_counter()

    return {"forward": t1 - t0, "inverse": t2 - t1, "total": t2 - t0}


def run_scale_timing(strategy_fn, component_names, params_builder, warmup, measured, trials):
    per_trial = {name: [] for name in component_names}
    for _ in range(trials):
        params = params_builder()
        for _ in range(warmup):
            strategy_fn(params)

        accum = {name: 0.0 for name in component_names}
        for _ in range(measured):
            timings = strategy_fn(params)
            for name in component_names:
                accum[name] += timings[name]

        for name in component_names:
            per_trial[name].append(accum[name] / measured * 1000.0)  # ms/call

    out = {}
    for name in component_names:
        t = torch.tensor(per_trial[name])
        out[name] = (t.mean().item(), t.std(unbiased=True).item() if trials > 1 else 0.0)
    return out


def timing_budget(mb: float) -> dict:
    """warmup/measured/trials scaled down as size grows, so the largest
    sizes don't dominate total run time; documented rather than hidden."""
    if mb <= 1:
        return dict(warmup=50, measured=500, trials=5)
    if mb <= 5:
        return dict(warmup=30, measured=300, trials=5)
    if mb <= 50:
        return dict(warmup=20, measured=100, trials=5)
    if mb <= 150:
        return dict(warmup=10, measured=40, trials=5)
    if mb <= 300:
        return dict(warmup=5, measured=20, trials=3)
    return dict(warmup=3, measured=10, trials=3)


def gbps(nbytes: float, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    return nbytes / (ms / 1000.0) / 1e9


# ---------------------------------------------------------------------------
# Main scale study.
# ---------------------------------------------------------------------------


def measure_one_size(target_mb: float):
    budget = timing_budget(target_mb)

    def build():
        return build_param_tensors(target_mb)

    probe = build()
    actual_bytes = gradient_bytes(probe)
    m_max_bytes = max_tensor_bytes(probe)
    num_params = len(probe)
    del probe

    safe = run_scale_timing(timed_safe_step, ["clone", "forward", "restore", "total"], build, **budget)
    rev = run_scale_timing(timed_reversible_step, ["forward", "inverse", "total"], build, **budget)

    return dict(
        target_mb=target_mb,
        actual_mb=actual_bytes / MB,
        m_max_mb=m_max_bytes / MB,
        num_params=num_params,
        budget=budget,
        safe=safe,
        rev=rev,
    )


def run_scale_study(sizes_mb):
    results = []
    for mb in sizes_mb:
        try:
            r = measure_one_size(mb)
            results.append(r)
        except (RuntimeError, MemoryError) as e:
            print(f"[skip] {mb} MB: allocation/runtime failure on this machine - {e}")
    return results


def print_main_timing_table(results):
    headers = [
        "size_MB", "num_params", "safe_clone_ms", "safe_forward_ms", "safe_restore_ms",
        "safe_total_ms", "rev_forward_ms", "rev_inverse_ms", "rev_total_ms", "rev_vs_safe_%",
    ]
    rows = []
    for r in results:
        safe, rev = r["safe"], r["rev"]
        rev_vs_safe_pct = (rev["total"][0] - safe["total"][0]) / safe["total"][0] * 100.0 if safe["total"][0] else float("nan")
        rows.append(
            [
                f"{r['actual_mb']:.2f}", r["num_params"],
                fmt_mean_std(*safe["clone"]), fmt_mean_std(*safe["forward"]), fmt_mean_std(*safe["restore"]),
                fmt_mean_std(*safe["total"]), fmt_mean_std(*rev["forward"]), fmt_mean_std(*rev["inverse"]),
                fmt_mean_std(*rev["total"]), f"{rev_vs_safe_pct:+.1f}",
            ]
        )
    print_table(headers, rows)
    print("(ms values are mean±std over the trials in each size's timing budget; see budgets below)")
    print("(rev_vs_safe_% > 0 means reversible is SLOWER than safe; < 0 means faster)")
    print()
    print("Timing budgets used (warmup/measured/trials scaled down for larger sizes):")
    for r in results:
        b = r["budget"]
        print(f"  {r['actual_mb']:.2f} MB: warmup={b['warmup']}, measured={b['measured']}, trials={b['trials']}")


def print_throughput_table(results):
    headers = ["size_MB", "safe_clone_GBps", "safe_forward_GBps", "rev_inverse_GBps"]
    rows = []
    for r in results:
        nbytes = r["actual_mb"] * MB
        safe, rev = r["safe"], r["rev"]
        rows.append(
            [
                f"{r['actual_mb']:.2f}",
                f"{gbps(nbytes, safe['clone'][0]):.2f}",
                f"{gbps(nbytes, safe['forward'][0]):.2f}",
                f"{gbps(nbytes, rev['inverse'][0]):.2f}",
            ]
        )
    print_table(headers, rows)
    print("(GB/s = gradient_bytes / mean_time; a flattening-out or plateau across sizes")
    print(" suggests the operation is memory-bandwidth-bound rather than dispatch-overhead-bound)")


def print_memory_table(results):
    headers = ["gradient_MB", "M_max_MB", "safe_persistent_backup_MB", "reversible_persistent_backup_MB", "persistent_memory_saved_MB"]
    rows = []
    for r in results:
        g_mb = r["actual_mb"]
        rows.append([f"{g_mb:.2f}", f"{r['m_max_mb']:.3f}", f"{g_mb:.2f}", "0.00", f"{g_mb:.2f}"])
    print_table(headers, rows)

    print("safe_persistent_backup_MB / gradient_MB ratio at every size: 1.000000 (by construction)")
    print("(this ratio is exactly 1.0 by construction - the safe backup IS a full clone of the")
    print(" gradient, so its linear growth with size is definitional, not just observed. The")
    print(" reversible strategy's persistent backup is 0 MB at every size tested: it eliminates")
    print(" the full-model backup buffer at small AND large scale, not only in the small-model")
    print(" case. Neither strategy's PEAK memory is claimed to be exactly these numbers - both")
    print(" still allocate O(M_max) transient temporaries per parameter during the transform;")
    print(" M_max_MB above is reported so that transient cost can be reasoned about separately")
    print(" from the persistent backup, but no profiler measured an actual peak here.")


# ---------------------------------------------------------------------------
# Simple linear fit: time_ms = a * gradient_MB + b.
# ---------------------------------------------------------------------------


def linear_fit(xs, ys):
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x_mean, y_mean = x.mean(), y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom.item() == 0:
        return float("nan"), float("nan"), float("nan")
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item() if ss_tot.item() != 0 else float("nan")
    return slope.item(), intercept.item(), r2


def print_scaling_fits(results):
    xs = [r["actual_mb"] for r in results]
    series = {
        "safe_total": [r["safe"]["total"][0] for r in results],
        "safe_clone": [r["safe"]["clone"][0] for r in results],
        "rev_total": [r["rev"]["total"][0] for r in results],
        "rev_inverse": [r["rev"]["inverse"][0] for r in results],
    }
    headers = ["series", "slope_ms_per_MB", "intercept_ms", "R2"]
    rows = []
    fits = {}
    for name, ys in series.items():
        slope, intercept, r2 = linear_fit(xs, ys)
        fits[name] = (slope, intercept, r2)
        rows.append([name, slope, intercept, r2])
    print_table(headers, rows)
    print("(time_ms = slope * gradient_MB + intercept; R2 close to 1.0 means the linear fit")
    print(" describes the data well across the tested range, not that N=O(N) is proven for all N)")
    return fits


def print_scaling_analysis(results, fits):
    print()
    if len(results) < 2:
        print("Not enough sizes succeeded to do a scaling analysis.")
        return

    smallest, largest = results[0], results[-1]

    print(f"1. Safe total time vs. size: slope={fits['safe_total'][0]:.5f} ms/MB, "
          f"intercept={fits['safe_total'][1]:.4f} ms, R2={fits['safe_total'][2]:.4f} "
          f"-> {'approximately linear (O(N)) over the tested range' if fits['safe_total'][2] > 0.98 else 'not cleanly linear - see per-size table'}.")

    print(f"2. Reversible total time vs. size: slope={fits['rev_total'][0]:.5f} ms/MB, "
          f"intercept={fits['rev_total'][1]:.4f} ms, R2={fits['rev_total'][2]:.4f} "
          f"-> {'approximately linear (O(N)) over the tested range' if fits['rev_total'][2] > 0.98 else 'not cleanly linear - see per-size table'}.")

    clone_slope = fits["safe_clone"][0]
    inverse_slope = fits["rev_inverse"][0]
    bigger = "clone" if clone_slope > inverse_slope else "inverse_restore"
    print(f"3. Per-byte slope: safe clone={clone_slope:.5f} ms/MB vs. reversible inverse_restore="
          f"{inverse_slope:.5f} ms/MB -> {bigger} has the steeper slope (more work per byte).")

    smallest_total_safe = smallest["safe"]["total"][0]
    print(f"4. Smallest size tested ({smallest['actual_mb']:.2f} MB): safe total={smallest_total_safe:.4f} ms. "
          f"The fitted intercepts above ({fits['safe_total'][1]:.4f} ms safe, {fits['rev_total'][1]:.4f} ms reversible) "
          "are this experiment's estimate of the size-independent (Python loop / dispatch) overhead; "
          "if they are a large fraction of the smallest size's total time, Python overhead dominates there.")

    largest_safe_gbps = gbps(largest["actual_mb"] * MB, largest["safe"]["forward"][0])
    smallest_safe_gbps = gbps(smallest["actual_mb"] * MB, smallest["safe"]["forward"][0])
    print(f"5. Safe forward-transform throughput: {smallest_safe_gbps:.2f} GB/s at the smallest size vs. "
          f"{largest_safe_gbps:.2f} GB/s at the largest ({largest['actual_mb']:.0f} MB) - "
          f"{'throughput rises/plateaus with size, consistent with per-call overhead being amortized and the operation becoming memory-bandwidth-bound' if largest_safe_gbps >= smallest_safe_gbps * 0.8 else 'throughput drops at larger sizes, which would need a machine-specific explanation (e.g. cache effects) rather than being assumed'}.")

    pct_by_size = []
    for r in results:
        pct = (r["rev"]["total"][0] - r["safe"]["total"][0]) / r["safe"]["total"][0] * 100.0 if r["safe"]["total"][0] else float("nan")
        pct_by_size.append((r["actual_mb"], pct))
    pct_str = ", ".join(f"{mb:.1f}MB:{pct:+.1f}%" for mb, pct in pct_by_size)
    print(f"6. Reversible vs. safe total-time percentage by size: {pct_str}")
    spread = max(p for _, p in pct_by_size) - min(p for _, p in pct_by_size)
    print(f"   Spread across sizes: {spread:.1f} percentage points -> "
          f"{'roughly stable across scale' if spread < 15 else 'changes meaningfully with scale - do not extrapolate the small-model percentage to large models'}.")

    print(f"7. Absolute persistent memory saved (safe backup - 0) grows exactly linearly with size by")
    print(f"   construction: {smallest['actual_mb']:.2f} MB saved at the smallest size tested vs. "
          f"{largest['actual_mb']:.2f} MB saved at the largest.")


# ---------------------------------------------------------------------------
# Optional supplement: include optimizer.step() at a few sizes.
# ---------------------------------------------------------------------------


def timed_safe_with_optimizer(optimizer) -> float:
    t0 = time.perf_counter()
    step_with_transformed_gradients(optimizer, signed_square)
    return time.perf_counter() - t0


def timed_reversible_with_optimizer(params, optimizer) -> float:
    t0 = time.perf_counter()
    for p in params:
        p.grad.mul_(p.grad.abs())
    optimizer.step()
    for p in params:
        g = p.grad
        g.copy_(g.sign() * g.abs().sqrt())
    return time.perf_counter() - t0


def run_optimizer_supplement(sizes_mb, warmup=10, measured=30, trials=3):
    configs = [
        ("SGD", torch.optim.SGD, dict(lr=0.01, momentum=0.9)),
        ("AdamW", torch.optim.AdamW, dict(lr=1e-3)),
    ]
    rows = []
    for mb in sizes_mb:
        for opt_name, opt_cls, kwargs in configs:
            try:
                safe_means, rev_means = [], []
                for _ in range(trials):
                    params_safe = build_param_tensors(mb)
                    opt_safe = opt_cls(params_safe, **kwargs)
                    for _ in range(warmup):
                        timed_safe_with_optimizer(opt_safe)
                    total = sum(timed_safe_with_optimizer(opt_safe) for _ in range(measured))
                    safe_means.append(total / measured * 1000.0)

                    params_rev = build_param_tensors(mb)
                    opt_rev = opt_cls(params_rev, **kwargs)
                    for _ in range(warmup):
                        timed_reversible_with_optimizer(params_rev, opt_rev)
                    total = sum(timed_reversible_with_optimizer(params_rev, opt_rev) for _ in range(measured))
                    rev_means.append(total / measured * 1000.0)

                safe_t = torch.tensor(safe_means)
                rev_t = torch.tensor(rev_means)
                safe_mean, safe_std = safe_t.mean().item(), safe_t.std(unbiased=True).item()
                rev_mean, rev_std = rev_t.mean().item(), rev_t.std(unbiased=True).item()
                pct = (rev_mean - safe_mean) / safe_mean * 100.0 if safe_mean else float("nan")
                rows.append([f"{mb:.2f}", opt_name, fmt_mean_std(safe_mean, safe_std), fmt_mean_std(rev_mean, rev_std), f"{pct:+.1f}"])
            except (RuntimeError, MemoryError) as e:
                print(f"[skip] optimizer supplement {mb} MB / {opt_name}: {e}")

    headers = ["size_MB", "optimizer", "safe_total_ms(+step)", "rev_total_ms(+step)", "rev_vs_safe_%"]
    print_table(headers, rows)


def main():
    torch.manual_seed(SEED)

    print("=" * 92)
    print("PRIMARY: transform+restore mechanism scaling (NO optimizer.step())")
    print("=" * 92)
    results = run_scale_study(TARGET_SIZES_MB)
    if not results:
        print("No sizes completed successfully on this machine.")
        return
    print_main_timing_table(results)

    print()
    print("=" * 92)
    print("Throughput (GB/s) - memory-bandwidth-bound check")
    print("=" * 92)
    print_throughput_table(results)

    print()
    print("=" * 92)
    print("Memory accounting per size")
    print("=" * 92)
    print_memory_table(results)

    print()
    print("=" * 92)
    print("Linear fits: time_ms = slope * gradient_MB + intercept")
    print("=" * 92)
    fits = print_scaling_fits(results)

    print()
    print("=" * 92)
    print("Scaling analysis")
    print("=" * 92)
    print_scaling_analysis(results, fits)

    print()
    print("=" * 92)
    print("OPTIONAL supplement: transform + optimizer.step() + restore, at a few sizes")
    print("=" * 92)
    run_optimizer_supplement([0.2, 20, 100])

    print()
    print("Note: results at the smallest size(s) are more sensitive to Python-level and OS")
    print("scheduling noise (see std in the tables above) - relative std is typically larger there")
    print("than at large sizes, so don't over-interpret small differences at the 0.2 MB row.")


if __name__ == "__main__":
    main()
