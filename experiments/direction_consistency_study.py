"""Research experiment (v2): does gradient DIRECTION predict stale-gradient
usefulness better than gradient AGE does?

This is a STANDALONE, OBSERVATION-ONLY research script. It does NOT change,
replace, or get imported by grad_transform.py, does not implement any
gradient reweighting (no ``w(c) * g``), does not modify any optimizer, and
does not build any staleness-mitigation algorithm. It only measures and
reports. If a second phase (e.g. gradient reweighting) is ever justified, it
belongs in a separate follow-up experiment.

Research question
------------------
Given a "stale" gradient computed at some historical model snapshot
theta_{t-k}, how well does it predict whether that gradient would still be
*useful* if applied at the CURRENT model theta_t?

We compare two candidate predictors of usefulness:

  1. staleness / age k  (how many learner steps old is the snapshot)
  2. directional consistency (cosine similarity between the stale gradient
     and a reference direction available at time t: either a reference
     gradient at theta_t, or an EMA of the trajectory's own past gradients)

v2 changes vs. v1, in response to two methodological problems found there
-------------------------------------------------------------------------
1. PREDICTOR/TARGET COUPLING: v1 computed both the direction predictor
   (cosine(g_stale, g_current)) AND the usefulness target (delta_loss from a
   probe update, evaluated at theta_t) using the SAME probe batch and the
   SAME "current" gradient. That let the two share statistical noise from
   one batch, and let the near-perfect direction-only F1 partly reflect a
   near-tautological algebraic relationship (delta_loss's sign is tied to
   the sign of a dot product that itself feeds the cosine). v2 fixes this by
   using THREE DISJOINT batches per observation:
     - B_stale: computes g_stale at theta_{t-k}
     - B_ref:   computes g_ref at theta_t (used only for the cosine predictor)
     - B_eval:  used ONLY to evaluate delta_loss after the probe update
   The direction predictor (cosine(g_stale, g_ref)) and the usefulness
   target (delta_loss on B_eval) now share no batch and no gradient tensor.
2. MAGNITUDE/DIRECTION CONFOUND: v1's probe update used the raw stale
   gradient, so a stale gradient with a larger norm produced a bigger probe
   step "for free" - conflating "good direction" with "big step". v2's
   PRIMARY probe update uses the L2-normalized direction
   g_dir = g_stale / (||g_stale||_2 + eps), so every stale gradient takes a
   step of the SAME size regardless of its raw magnitude. The raw-gradient
   update from v1 is kept as a labeled SECONDARY/CONTROL condition, so we
   can directly check whether normalizing changes the conclusion.

IMPORTANT - what "staleness" means here
----------------------------------------
This experiment simulates staleness with a CONTROLLED PROXY: we snapshot a
single learner's own parameters during ordinary synchronous training, and
later recompute gradients at an old snapshot on fixed, held-out probe
batches. This is NOT a model of real asynchronous / distributed rollout
staleness (no actor lag, no partial rollouts, no RLVR, no distributed
queueing). Conclusions here are a hypothesis input for later, more realistic
studies - not a final claim about production async training.

Explicitly OUT OF SCOPE for this script (see README.md's "Conceptual scope"
for the sibling scope note on grad_transform.py):
  - w(c) * g gradient reweighting of any kind
  - optimizer modification
  - real async rollout infrastructure, RLVR, distributed training, FSDP/ZeRO
  - gradient sketching / low-rank approximation
  - large models or GPU (this is a tiny CPU-only MLP)

Run with:  python experiments/direction_consistency_study.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 0

# --- synthetic problem size (kept tiny on purpose: CPU-only, observation study) ---
IN_DIM = 16
HIDDEN_DIM = 32
OUT_DIM = 1
TRAIN_N = 2000
TRAIN_BATCH = 32
PROBE_N = 256  # size of EACH of the three disjoint probe sets (B_stale, B_ref, B_eval)
NOISE_STD = 0.05

# --- training trajectory length / snapshot grid ---
TOTAL_STEPS = 400
K_LIST = [1, 2, 5, 10, 20, 50]
T_START = max(K_LIST) + 10  # margin so every k in K_LIST is valid for every t we test
SGD_T_STRIDE = 2
ADAMW_T_STRIDE = 5

# --- probe-update step sizes for the usefulness proxy (delta_loss). ---
# PRIMARY: applied to the L2-NORMALIZED stale direction g_dir (same step
# size for every observation, regardless of the stale gradient's raw norm).
ETA_PROBE_NORMALIZED_LIST = [1e-3, 1e-2, 5e-2]
ETA_PROBE_NORM_DEFAULT = ETA_PROBE_NORMALIZED_LIST[0]
# SECONDARY / CONTROL: applied to the RAW stale gradient (this is what v1
# did). Kept only to check whether normalizing the step changes the
# conclusion, not as the headline result.
ETA_PROBE_RAW_LIST = [1e-4, 1e-3, 5e-3]
ETA_PROBE_RAW_DEFAULT = ETA_PROBE_RAW_LIST[0]

# Reference point from the v1 (coupled-batch, raw-magnitude) run, quoted
# only for comparison in the final-answers section - NOT used anywhere in
# the v2 computation itself.
V1_DIRECTION_ONLY_BEST_F1 = 1.0000
V1_AGE_ONLY_BEST_F1 = 0.7676

# --- EMA betas for the "historical trajectory direction" signal ---
EMA_BETAS = [0.9, 0.99]

EPS = 1e-12

COSINE_BINS = [(-1.0, -0.5), (-0.5, 0.0), (0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


# ---------------------------------------------------------------------------
# Small table printer (no external dependencies) - same convention as the
# other experiments in this repo (see experiments/reversible_tradeoff.py).
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
# Dependency-free correlation helpers (no scipy in this environment).
# Pearson is the standard linear-correlation formula; Spearman is Pearson
# computed on ranks (average rank for ties), which does not assume a linear
# relationship - preferred here since delta_loss vs. age/cosine need not be
# linear.
# ---------------------------------------------------------------------------


def pearson_corr(xs, ys) -> float:
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    if x.numel() < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = torch.sqrt((xm * xm).sum() * (ym * ym).sum())
    if denom.item() == 0.0:
        return float("nan")
    return ((xm * ym).sum() / denom).item()


def _rankdata(xs) -> torch.Tensor:
    x = torch.tensor(xs, dtype=torch.float64)
    n = x.numel()
    order = torch.argsort(x)
    sorted_x = x[order]
    ranks_sorted = torch.empty(n, dtype=torch.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank over the tie block
        ranks_sorted[i : j + 1] = avg_rank
        i = j + 1
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearman_corr(xs, ys) -> float:
    if len(xs) < 2:
        return float("nan")
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    return pearson_corr(rx.tolist(), ry.tolist())


# ---------------------------------------------------------------------------
# Gradient-vector math.
# ---------------------------------------------------------------------------


def flatten(tensors) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_like(flat: torch.Tensor, like_tensors):
    """Inverse of flatten(): split `flat` back into per-parameter tensors
    matching the shapes of `like_tensors`, in the same order."""
    out = []
    idx = 0
    for t in like_tensors:
        n = t.numel()
        out.append(flat[idx : idx + n].reshape(t.shape))
        idx += n
    return out


def l2_normalize(g: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """g_hat = g / (||g||_2 + eps). True L2-normalized direction, NOT g/|g| (sign)."""
    return g / (g.norm() + eps)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = EPS) -> float:
    denom = a.norm() * b.norm() + eps
    return (torch.dot(a, b) / denom).item()


# ---------------------------------------------------------------------------
# Synthetic regression problem: a small fixed "teacher" MLP (different width
# than the learner) generates y = teacher(x) + noise. The learner then has a
# real, non-trivial signal to fit but cannot represent it exactly, so
# training dynamics stay interesting (gradients keep changing direction)
# rather than collapsing to ~0 within a handful of steps.
# ---------------------------------------------------------------------------


def build_teacher() -> nn.Module:
    g = torch.Generator().manual_seed(SEED + 1000)
    teacher = nn.Sequential(nn.Linear(IN_DIM, 24), nn.Tanh(), nn.Linear(24, OUT_DIM))
    with torch.no_grad():
        for p in teacher.parameters():
            p.copy_(torch.randn(p.shape, generator=g))
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def make_dataset(n: int, seed: int, teacher: nn.Module):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_DIM, generator=g)
    with torch.no_grad():
        y = teacher(x) + NOISE_STD * torch.randn(n, OUT_DIM, generator=g)
    return x, y


def build_model() -> nn.Module:
    torch.manual_seed(SEED)
    return nn.Sequential(nn.Linear(IN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, OUT_DIM))


def param_names(model: nn.Module):
    return [name for name, _ in model.named_parameters()]


def clone_state_dict(model: nn.Module):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def compute_grad(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    """Returns (per-parameter grad list matching model.parameters() order, loss value)."""
    model.zero_grad(set_to_none=True)
    loss = nn.functional.mse_loss(model(x), y)
    loss.backward()
    grads = [p.grad.detach().clone() for p in model.parameters()]
    return grads, loss.item()


def loss_only(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return nn.functional.mse_loss(model(x), y).item()


def probe_update_loss(model: nn.Module, base_state, grad_list, eta: float, x, y) -> float:
    """L(theta_t - eta * grad_list), leaving `model` reset to base_state afterwards."""
    model.load_state_dict(base_state)
    with torch.no_grad():
        for p, g in zip(model.parameters(), grad_list):
            p.sub_(eta * g)
    loss = loss_only(model, x, y)
    model.load_state_dict(base_state)
    return loss


# ---------------------------------------------------------------------------
# Observation record.
# ---------------------------------------------------------------------------


@dataclass
class Observation:
    optimizer: str
    t: int
    k: int
    stale_grad_norm: float          # ||g_stale||, computed on B_stale
    ref_grad_norm: float            # ||g_ref||, computed on B_ref
    cosine_stale_ref: float         # cos(g_stale, g_ref) - the direction predictor
    cosine_stale_ema: dict          # beta -> cos(g_stale, EMA historical direction)
    dot_ref_stale: float            # raw dot(g_ref, g_stale) - diagnostic only
    predicted_first_order_delta: float  # eta_raw_default * dot(g_eval, g_stale); g_eval is B_eval's own gradient (decoupled from cosine)
    delta_loss_norm: dict           # eta -> delta_loss, PRIMARY: probe update uses normalized g_dir, evaluated on B_eval
    delta_loss_raw: dict            # eta -> delta_loss, CONTROL: probe update uses raw g_stale, evaluated on B_eval
    delta_loss_default: float       # = delta_loss_norm[ETA_PROBE_NORM_DEFAULT]  (PRIMARY usefulness proxy)
    delta_loss_raw_default: float   # = delta_loss_raw[ETA_PROBE_RAW_DEFAULT]    (CONTROL usefulness proxy)
    useful: bool                    # delta_loss_default > 0        (PRIMARY label)
    useful_raw: bool                # delta_loss_raw_default > 0    (CONTROL label)
    layer_cosine: dict = field(default_factory=dict)  # param_name -> cos(g_stale_layer, g_ref_layer)


# ---------------------------------------------------------------------------
# PART A: train the learner, saving a dense trajectory of parameter
# snapshots and of the running EMA gradient-direction, so both are available
# "as of step t" without ever looking into the future. EMA uses the
# learner's own TRAINING-batch gradients (not any of the three held-out
# probe sets), so it introduces no coupling with the usefulness target.
# ---------------------------------------------------------------------------


def train_and_collect_snapshots(optimizer_cls, optimizer_kwargs, x_train, y_train, total_steps: int):
    model = build_model()
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    names = param_names(model)

    snapshots = {0: clone_state_dict(model)}
    ema = {beta: torch.zeros(sum(p.numel() for p in model.parameters())) for beta in EMA_BETAS}
    ema_snapshots = {}  # t -> {beta: tensor}, defined for t >= 1
    train_losses = []

    batch_gen = torch.Generator().manual_seed(SEED + 2000)

    for step in range(1, total_steps + 1):
        idx = torch.randint(0, x_train.shape[0], (TRAIN_BATCH,), generator=batch_gen)
        xb, yb = x_train[idx], y_train[idx]

        grad_list, loss_val = compute_grad(model, xb, yb)
        train_losses.append(loss_val)

        g_hat = l2_normalize(flatten(grad_list))
        for beta in EMA_BETAS:
            ema[beta] = beta * ema[beta] + (1 - beta) * g_hat
        ema_snapshots[step] = {beta: ema[beta].clone() for beta in EMA_BETAS}

        optimizer.step()
        snapshots[step] = clone_state_dict(model)

    return dict(
        model=model,
        names=names,
        snapshots=snapshots,
        ema_snapshots=ema_snapshots,
        train_losses=train_losses,
    )


# ---------------------------------------------------------------------------
# PART B: build (t, k) diagnostic observations from a trained trajectory,
# using THREE DISJOINT probe sets (B_stale, B_ref, B_eval).
# ---------------------------------------------------------------------------


def build_observations(
    optimizer_name: str,
    trajectory: dict,
    x_stale, y_stale,
    x_ref, y_ref,
    x_eval, y_eval,
    t_start: int,
    t_stride: int,
):
    snapshots = trajectory["snapshots"]
    ema_snapshots = trajectory["ema_snapshots"]
    names = trajectory["names"]
    total_steps = max(snapshots.keys())

    probe_model = build_model()  # architecture only; state is overwritten before every use

    observations: list[Observation] = []
    ref_cache = {}  # t -> (ref_grads, ref_flat, eval_loss_baseline, eval_flat)

    t_values = list(range(t_start, total_steps + 1, t_stride))
    for t in t_values:
        if t not in ref_cache:
            probe_model.load_state_dict(snapshots[t])
            ref_grads, _ref_loss_on_bref = compute_grad(probe_model, x_ref, y_ref)
            ref_flat = flatten(ref_grads)

            probe_model.load_state_dict(snapshots[t])
            eval_grads, eval_loss_baseline = compute_grad(probe_model, x_eval, y_eval)
            eval_flat = flatten(eval_grads)

            ref_cache[t] = (ref_grads, ref_flat, eval_loss_baseline, eval_flat)
        ref_grads, ref_flat, eval_loss_baseline, eval_flat = ref_cache[t]

        for k in K_LIST:
            if t - k < 0:
                continue
            probe_model.load_state_dict(snapshots[t - k])
            stale_grads, _stale_loss_on_bstale = compute_grad(probe_model, x_stale, y_stale)
            stale_flat = flatten(stale_grads)

            cos_ref = cosine(stale_flat, ref_flat)
            cos_ema = {beta: cosine(stale_flat, ema_snapshots[t][beta]) for beta in EMA_BETAS}
            dot_rs = torch.dot(ref_flat, stale_flat).item()

            g_dir_flat = l2_normalize(stale_flat)
            g_dir_list = unflatten_like(g_dir_flat, stale_grads)

            delta_norm = {}
            for eta in ETA_PROBE_NORMALIZED_LIST:
                loss_after = probe_update_loss(probe_model, snapshots[t], g_dir_list, eta, x_eval, y_eval)
                delta_norm[eta] = eval_loss_baseline - loss_after

            delta_raw = {}
            for eta in ETA_PROBE_RAW_LIST:
                loss_after = probe_update_loss(probe_model, snapshots[t], stale_grads, eta, x_eval, y_eval)
                delta_raw[eta] = eval_loss_baseline - loss_after

            # First-order Taylor check: L(theta_t - eta*g_stale) ~= L(theta_t) - eta*(g_eval . g_stale),
            # where g_eval is B_eval's OWN gradient at theta_t (not g_ref) - this keeps the check
            # decoupled from the cosine predictor while still being a valid first-order approximation
            # of the RAW-update delta_loss on B_eval.
            predicted_fo = ETA_PROBE_RAW_DEFAULT * torch.dot(eval_flat, stale_flat).item()

            layer_cos = {
                name: cosine(gs.reshape(-1), gr.reshape(-1))
                for name, gs, gr in zip(names, stale_grads, ref_grads)
            }

            observations.append(
                Observation(
                    optimizer=optimizer_name,
                    t=t,
                    k=k,
                    stale_grad_norm=stale_flat.norm().item(),
                    ref_grad_norm=ref_flat.norm().item(),
                    cosine_stale_ref=cos_ref,
                    cosine_stale_ema=cos_ema,
                    dot_ref_stale=dot_rs,
                    predicted_first_order_delta=predicted_fo,
                    delta_loss_norm=delta_norm,
                    delta_loss_raw=delta_raw,
                    delta_loss_default=delta_norm[ETA_PROBE_NORM_DEFAULT],
                    delta_loss_raw_default=delta_raw[ETA_PROBE_RAW_DEFAULT],
                    useful=delta_norm[ETA_PROBE_NORM_DEFAULT] > 0,
                    useful_raw=delta_raw[ETA_PROBE_RAW_DEFAULT] > 0,
                    layer_cosine=layer_cos,
                )
            )

    return observations


# ---------------------------------------------------------------------------
# ANALYSIS 1: age vs. usefulness.
# ---------------------------------------------------------------------------


def analysis_age_vs_usefulness(obs: list[Observation]):
    print("-- mean delta_loss / useful-fraction by age k --")
    rows = []
    for k in K_LIST:
        sub = [o for o in obs if o.k == k]
        if not sub:
            continue
        dl = [o.delta_loss_default for o in sub]
        useful_frac = sum(o.useful for o in sub) / len(sub)
        rows.append([k, len(sub), sum(dl) / len(dl), useful_frac, min(dl), max(dl)])
    print_table(["age_k", "n", "mean_delta_loss", "useful_fraction", "min_delta_loss", "max_delta_loss"], rows)

    ages = [o.k for o in obs]
    dls = [o.delta_loss_default for o in obs]
    pear = pearson_corr(ages, dls)
    spear = spearman_corr(ages, dls)
    print()
    print(f"correlation(age, delta_loss): pearson={pear:.4f}  spearman={spear:.4f}  (n={len(obs)})")
    return dict(pearson=pear, spearman=spear)


# ---------------------------------------------------------------------------
# ANALYSIS 2: direction consistency vs. usefulness.
# ---------------------------------------------------------------------------


def _bin_label(lo, hi):
    return f"[{lo:.2f},{hi:.2f}]"


def analysis_direction_vs_usefulness(obs: list[Observation], cosine_field: str = "cosine_stale_ref"):
    def get_cos(o):
        if cosine_field == "cosine_stale_ref":
            return o.cosine_stale_ref
        beta = float(cosine_field.split("_")[-1])
        return o.cosine_stale_ema[beta]

    coss = [get_cos(o) for o in obs]
    dls = [o.delta_loss_default for o in obs]
    pear = pearson_corr(coss, dls)
    spear = spearman_corr(coss, dls)
    print(f"[{cosine_field}] correlation with delta_loss: pearson={pear:.4f}  spearman={spear:.4f}  (n={len(obs)})")

    rows = []
    for lo, hi in COSINE_BINS:
        sub = [o for o, c in zip(obs, coss) if (c > lo or (lo == -1.0 and c >= lo)) and c <= hi]
        if not sub:
            rows.append([_bin_label(lo, hi), 0, float("nan"), float("nan")])
            continue
        sub_dl = [o.delta_loss_default for o in sub]
        useful_frac = sum(o.useful for o in sub) / len(sub)
        rows.append([_bin_label(lo, hi), len(sub), sum(sub_dl) / len(sub_dl), useful_frac])
    print_table(["cosine_bin", "n", "mean_delta_loss", "useful_fraction"], rows)
    return dict(pearson=pear, spearman=spear)


# ---------------------------------------------------------------------------
# ANALYSIS 3: age-only vs. direction-only vs. combined, as USEFUL-GRADIENT
# classifiers (threshold sweeps, precision / recall / F1 - no ML model).
# `useful_fn` / `cos_fn` are parameterized so the same machinery can score
# the PRIMARY (normalized, decoupled) label and the CONTROL (raw) label.
# ---------------------------------------------------------------------------


def _prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision == precision and recall == recall and (precision + recall) > 0) else float("nan")
    return precision, recall, f1


def _sweep_age(obs: list[Observation], useful_fn=lambda o: o.useful):
    thresholds = [0] + sorted(set(K_LIST))
    rows = []
    best = None
    for K in thresholds:
        tp = fp = fn = 0
        for o in obs:
            pred_pos = o.k <= K
            u = useful_fn(o)
            if pred_pos and u:
                tp += 1
            elif pred_pos and not u:
                fp += 1
            elif (not pred_pos) and u:
                fn += 1
        p, r, f1 = _prf1(tp, fp, fn)
        rows.append([K, tp, fp, fn, p, r, f1])
        if f1 == f1 and (best is None or f1 > best[1]):
            best = (K, f1, p, r)
    return rows, best


def _sweep_cosine(obs: list[Observation], get_cos, useful_fn=lambda o: o.useful):
    thresholds = [round(-1.0 + 0.1 * i, 2) for i in range(21)]  # -1.0 .. 1.0 step 0.1
    rows = []
    best = None
    for C in thresholds:
        tp = fp = fn = 0
        for o in obs:
            pred_pos = get_cos(o) >= C
            u = useful_fn(o)
            if pred_pos and u:
                tp += 1
            elif pred_pos and not u:
                fp += 1
            elif (not pred_pos) and u:
                fn += 1
        p, r, f1 = _prf1(tp, fp, fn)
        rows.append([C, tp, fp, fn, p, r, f1])
        if f1 == f1 and (best is None or f1 > best[1]):
            best = (C, f1, p, r)
    return rows, best


def quick_best_f1s(obs: list[Observation], useful_fn):
    """Age-only and cosine-only best F1, without printing full sweep tables -
    used for the raw-vs-normalized side-by-side comparison."""
    _, age_best = _sweep_age(obs, useful_fn=useful_fn)
    _, cos_best = _sweep_cosine(obs, lambda o: o.cosine_stale_ref, useful_fn=useful_fn)
    return age_best, cos_best


def analysis_predictive_power(obs: list[Observation]):
    get_cos = lambda o: o.cosine_stale_ref

    age_rows, age_best = _sweep_age(obs)
    print("-- age-based classifier sweep: predict useful iff age <= K --")
    print_table(["K", "tp", "fp", "fn", "precision", "recall", "f1"], age_rows)
    print(f"best age-based: K={age_best[0]}  F1={age_best[1]:.4f}  precision={age_best[2]:.4f}  recall={age_best[3]:.4f}")

    print()
    cos_rows, cos_best = _sweep_cosine(obs, get_cos)
    print("-- direction-based classifier sweep: predict useful iff cosine_stale_ref >= C --")
    print_table(["C", "tp", "fp", "fn", "precision", "recall", "f1"], cos_rows)
    print(f"best direction-based: C={cos_best[0]}  F1={cos_best[1]:.4f}  precision={cos_best[2]:.4f}  recall={cos_best[3]:.4f}")

    print()
    print("-- combined classifier grid search: predict useful iff (age <= K) AND (cosine >= C) --")
    best_combo = None
    combo_rows = []
    for K in [0] + sorted(set(K_LIST)):
        for C in [round(-1.0 + 0.2 * i, 2) for i in range(11)]:  # coarser grid, 11 values
            tp = fp = fn = 0
            for o in obs:
                pred_pos = (o.k <= K) and (get_cos(o) >= C)
                if pred_pos and o.useful:
                    tp += 1
                elif pred_pos and not o.useful:
                    fp += 1
                elif (not pred_pos) and o.useful:
                    fn += 1
            p, r, f1 = _prf1(tp, fp, fn)
            combo_rows.append([K, C, tp, fp, fn, p, r, f1])
            if f1 == f1 and (best_combo is None or f1 > best_combo[2]):
                best_combo = (K, C, f1, p, r)
    top5 = sorted([r for r in combo_rows if r[-1] == r[-1]], key=lambda r: r[-1], reverse=True)[:5]
    print_table(["K", "C", "tp", "fp", "fn", "precision", "recall", "f1"], top5)
    print(f"best combined: K={best_combo[0]}  C={best_combo[1]}  F1={best_combo[2]:.4f}  "
          f"precision={best_combo[3]:.4f}  recall={best_combo[4]:.4f}")

    print()
    print(f"Summary — best F1: age-only={age_best[1]:.4f}  direction-only={cos_best[1]:.4f}  combined={best_combo[2]:.4f}")
    print(f"(for reference, the v1 run - coupled batch, raw-magnitude probe update - had "
          f"age-only F1={V1_AGE_ONLY_BEST_F1:.4f}, direction-only F1={V1_DIRECTION_ONLY_BEST_F1:.4f})")

    return dict(age_best=age_best, cos_best=cos_best, combo_best=best_combo)


# ---------------------------------------------------------------------------
# ANALYSIS: raw (v1-style) vs. normalized (v2 primary) probe update - does
# removing the magnitude confound change the conclusion?
# ---------------------------------------------------------------------------


def analysis_raw_vs_normalized(obs: list[Observation]):
    dls_norm = [o.delta_loss_default for o in obs]
    dls_raw = [o.delta_loss_raw_default for o in obs]

    corr_norm_raw = pearson_corr(dls_norm, dls_raw)
    sign_agree = sum((d1 > 0) == (d2 > 0) for d1, d2 in zip(dls_norm, dls_raw)) / len(obs)
    print(f"correlation(delta_loss_normalized, delta_loss_raw): {corr_norm_raw:.4f}")
    print(f"sign agreement between 'useful' (normalized) and 'useful_raw' (raw) labels: {sign_agree:.4f}")

    cos_corr_norm = pearson_corr([o.cosine_stale_ref for o in obs], dls_norm)
    cos_corr_raw = pearson_corr([o.cosine_stale_ref for o in obs], dls_raw)
    print(f"\ncorrelation(cosine_stale_ref, delta_loss): normalized={cos_corr_norm:.4f}  raw={cos_corr_raw:.4f}")

    age_best_norm, cos_best_norm = quick_best_f1s(obs, lambda o: o.useful)
    age_best_raw, cos_best_raw = quick_best_f1s(obs, lambda o: o.useful_raw)
    rows = [
        ["normalized (primary)", age_best_norm[1], cos_best_norm[1]],
        ["raw (control, v1-style step)", age_best_raw[1], cos_best_raw[1]],
    ]
    print()
    print_table(["probe-update variant", "best age-only F1", "best direction-only F1"], rows)

    return dict(
        corr_norm_raw=corr_norm_raw,
        sign_agree=sign_agree,
        age_f1_norm=age_best_norm[1],
        cos_f1_norm=cos_best_norm[1],
        age_f1_raw=age_best_raw[1],
        cos_f1_raw=cos_best_raw[1],
    )


# ---------------------------------------------------------------------------
# ANALYSIS 4: same-age variance - is age alone sufficient, or does cosine
# still separate helpful from harmful gradients AT A FIXED age?
# ---------------------------------------------------------------------------


def analysis_same_age_variance(obs: list[Observation]):
    rows = []
    contrast_found = False
    for k in K_LIST:
        sub = [o for o in obs if o.k == k]
        if len(sub) < 2:
            continue
        coss = [o.cosine_stale_ref for o in sub]
        dls = [o.delta_loss_default for o in sub]
        cos_std = torch.tensor(coss).std(unbiased=True).item()
        dl_std = torch.tensor(dls).std(unbiased=True).item()
        within_age_corr = pearson_corr(coss, dls)
        rows.append([k, len(sub), min(coss), max(coss), cos_std, dl_std, within_age_corr])

        best_o = max(sub, key=lambda o: o.cosine_stale_ref)
        worst_o = min(sub, key=lambda o: o.cosine_stale_ref)
        if (best_o.useful) != (worst_o.useful):
            contrast_found = True
            print(
                f"  age={k}: sample A cosine={best_o.cosine_stale_ref:.3f}, "
                f"delta_loss={best_o.delta_loss_default:+.3e} ({'useful' if best_o.useful else 'harmful'})  |  "
                f"sample B cosine={worst_o.cosine_stale_ref:.3f}, "
                f"delta_loss={worst_o.delta_loss_default:+.3e} ({'useful' if worst_o.useful else 'harmful'})"
            )

    print("-- within-age spread of cosine and delta_loss --")
    print_table(
        ["age_k", "n", "cos_min", "cos_max", "cos_std", "delta_loss_std", "within_age_corr(cos,delta_loss)"],
        rows,
    )
    if not contrast_found:
        print("(no fixed-age case in this run had cosine's max/min sample land on opposite sides of usefulness)")
    return dict(contrast_found=contrast_found)


# ---------------------------------------------------------------------------
# ANALYSIS 5: "older but aligned" vs. "younger but conflicting" - the most
# direct evidence for/against "older necessarily means worse".
# ---------------------------------------------------------------------------


def analysis_old_aligned_vs_young_conflicting(obs: list[Observation], top_n: int = 5):
    old_k = {k for k in K_LIST if k >= 20}
    young_k = {k for k in K_LIST if k <= 2}

    old_aligned = [o for o in obs if o.k in old_k and o.cosine_stale_ref > 0.5 and o.delta_loss_default > 0]
    young_conflicting = [o for o in obs if o.k in young_k and o.cosine_stale_ref < 0.0 and o.delta_loss_default < 0]

    old_aligned.sort(key=lambda o: (o.k, o.cosine_stale_ref, o.delta_loss_default), reverse=True)
    young_conflicting.sort(key=lambda o: (o.cosine_stale_ref, o.delta_loss_default))

    print(f"'old but aligned' candidates (k in {sorted(old_k)}, cosine>0.5, delta_loss>0): {len(old_aligned)} found")
    if old_aligned:
        rows = [[o.t, o.k, f"{o.cosine_stale_ref:.3f}", f"{o.delta_loss_default:+.3e}"] for o in old_aligned[:top_n]]
        print_table(["t", "k", "cosine_stale_ref", "delta_loss"], rows)

    print()
    print(f"'young but conflicting' candidates (k in {sorted(young_k)}, cosine<0, delta_loss<0): {len(young_conflicting)} found")
    if young_conflicting:
        rows = [[o.t, o.k, f"{o.cosine_stale_ref:.3f}", f"{o.delta_loss_default:+.3e}"] for o in young_conflicting[:top_n]]
        print_table(["t", "k", "cosine_stale_ref", "delta_loss"], rows)

    if not old_aligned and not young_conflicting:
        print()
        print("(neither pattern was observed in this run - not manufactured; reporting the absence honestly)")

    return dict(n_old_aligned=len(old_aligned), n_young_conflicting=len(young_conflicting))


# ---------------------------------------------------------------------------
# LAYER-WISE extension (observation only, no reweighting).
# ---------------------------------------------------------------------------


def analysis_layerwise(obs: list[Observation], names):
    print("-- per-layer directional consistency summary (observation only) --")
    rows = []
    masked_conflict_count = 0
    for name in names:
        layer_coss = [o.layer_cosine[name] for o in obs]
        global_coss = [o.cosine_stale_ref for o in obs]
        corr_with_global = pearson_corr(layer_coss, global_coss)
        t = torch.tensor(layer_coss)
        rows.append([name, t.mean().item(), t.std(unbiased=True).item(), t.min().item(), t.max().item(), corr_with_global])

    print_table(["layer", "mean_cosine", "std_cosine", "min_cosine", "max_cosine", "corr_with_global_cosine"], rows)

    for o in obs:
        if o.cosine_stale_ref > 0.3 and any(o.layer_cosine[name] < -0.3 for name in names):
            masked_conflict_count += 1
    frac = masked_conflict_count / len(obs) if obs else float("nan")
    print(f"\nObservations where global cosine > 0.3 but at least one layer's cosine < -0.3: "
          f"{masked_conflict_count}/{len(obs)} ({frac * 100:.2f}%)")
    print("(this would indicate the global/flattened cosine can mask per-layer conflict; observation only)")
    return dict(masked_conflict_fraction=frac)


# ---------------------------------------------------------------------------
# EMA direction study (kept as a supplementary signal per the v2 spec).
# ---------------------------------------------------------------------------


def analysis_ema_study(obs: list[Observation]):
    print("-- stability: std of cosine signal within each age bucket (ref-grad vs. EMA) --")
    rows = []
    for k in K_LIST:
        sub = [o for o in obs if o.k == k]
        if not sub:
            continue
        cur_std = torch.tensor([o.cosine_stale_ref for o in sub]).std(unbiased=True).item()
        row = [k, len(sub), cur_std]
        for beta in EMA_BETAS:
            ema_std = torch.tensor([o.cosine_stale_ema[beta] for o in sub]).std(unbiased=True).item()
            row.append(ema_std)
        rows.append(row)
    print_table(["age_k", "n", "std_cos_ref"] + [f"std_cos_ema_{b}" for b in EMA_BETAS], rows)

    print()
    print("-- predictive power: correlation(signal, delta_loss), split by recent (k<=5) vs. old (k>=20) --")
    rows2 = []
    for label, pred in [("all", lambda o: True), ("k<=5 (recent)", lambda o: o.k <= 5), ("k>=20 (old)", lambda o: o.k >= 20)]:
        sub = [o for o in obs if pred(o)]
        if not sub:
            continue
        dls = [o.delta_loss_default for o in sub]
        row = [label, len(sub), pearson_corr([o.cosine_stale_ref for o in sub], dls)]
        for beta in EMA_BETAS:
            row.append(pearson_corr([o.cosine_stale_ema[beta] for o in sub], dls))
        rows2.append(row)
    print_table(["subset", "n", "corr_cos_ref"] + [f"corr_cos_ema_{b}" for b in EMA_BETAS], rows2)
    return dict()


# ---------------------------------------------------------------------------
# eta_probe robustness check (run separately for the normalized-primary and
# raw-control probe-update families).
# ---------------------------------------------------------------------------


def analysis_eta_sensitivity(obs: list[Observation]):
    print(f"-- PRIMARY (normalized step) sensitivity to eta_probe (values tested: {ETA_PROBE_NORMALIZED_LIST}) --")
    rows = []
    base_signs = [o.delta_loss_norm[ETA_PROBE_NORM_DEFAULT] > 0 for o in obs]
    for eta in ETA_PROBE_NORMALIZED_LIST:
        dls = [o.delta_loss_norm[eta] for o in obs]
        signs = [d > 0 for d in dls]
        agree = sum(a == b for a, b in zip(signs, base_signs)) / len(obs)
        corr_with_default = pearson_corr([o.delta_loss_norm[ETA_PROBE_NORM_DEFAULT] for o in obs], dls)
        useful_frac = sum(signs) / len(obs)
        rows.append([eta, useful_frac, agree, corr_with_default])
    print_table(
        ["eta_probe", "useful_fraction", "sign_agreement_with_default", f"corr_with_delta_loss(eta={ETA_PROBE_NORM_DEFAULT})"],
        rows,
    )

    print()
    print(f"-- CONTROL (raw step) sensitivity to eta_probe (values tested: {ETA_PROBE_RAW_LIST}) --")
    rows_raw = []
    base_signs_raw = [o.delta_loss_raw[ETA_PROBE_RAW_DEFAULT] > 0 for o in obs]
    for eta in ETA_PROBE_RAW_LIST:
        dls = [o.delta_loss_raw[eta] for o in obs]
        signs = [d > 0 for d in dls]
        agree = sum(a == b for a, b in zip(signs, base_signs_raw)) / len(obs)
        corr_with_default = pearson_corr([o.delta_loss_raw[ETA_PROBE_RAW_DEFAULT] for o in obs], dls)
        useful_frac = sum(signs) / len(obs)
        rows_raw.append([eta, useful_frac, agree, corr_with_default])
    print_table(
        ["eta_probe", "useful_fraction", "sign_agreement_with_default", f"corr_with_delta_loss(eta={ETA_PROBE_RAW_DEFAULT})"],
        rows_raw,
    )

    fo_corr = pearson_corr(
        [o.predicted_first_order_delta for o in obs], [o.delta_loss_raw_default for o in obs]
    )
    print(f"\ncorrelation(first-order predicted delta_loss [uses B_eval's OWN gradient, decoupled "
          f"from cosine], actual delta_loss_raw @ eta={ETA_PROBE_RAW_DEFAULT}): {fo_corr:.4f}")
    return dict()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_for_optimizer(
    name, optimizer_cls, optimizer_kwargs,
    x_train, y_train,
    x_stale, y_stale, x_ref, y_ref, x_eval, y_eval,
    t_stride,
):
    torch.manual_seed(SEED)
    trajectory = train_and_collect_snapshots(optimizer_cls, optimizer_kwargs, x_train, y_train, TOTAL_STEPS)
    losses = trajectory["train_losses"]
    print(f"[{name}] training-batch loss: step1={losses[0]:.4f}  step{len(losses)//4}={losses[len(losses)//4]:.4f}  "
          f"step{len(losses)//2}={losses[len(losses)//2]:.4f}  step{len(losses)}={losses[-1]:.4f}")
    obs = build_observations(
        name, trajectory, x_stale, y_stale, x_ref, y_ref, x_eval, y_eval, T_START, t_stride
    )
    print(f"[{name}] collected {len(obs)} (t, k) observations "
          f"(t in [{T_START}, {TOTAL_STEPS}] stride {t_stride}, k in {K_LIST})")
    return trajectory, obs


def main():
    torch.manual_seed(SEED)
    teacher = build_teacher()
    x_train, y_train = make_dataset(TRAIN_N, seed=SEED + 10, teacher=teacher)
    # Three DISJOINT probe sets (different seeds -> statistically independent draws):
    x_stale, y_stale = make_dataset(PROBE_N, seed=SEED + 20, teacher=teacher)
    x_ref, y_ref = make_dataset(PROBE_N, seed=SEED + 21, teacher=teacher)
    x_eval, y_eval = make_dataset(PROBE_N, seed=SEED + 22, teacher=teacher)

    print("=" * 92)
    print("SETUP (v2)")
    print("=" * 92)
    print(f"model: Linear({IN_DIM},{HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM},{OUT_DIM}), "
          f"{sum(p.numel() for p in build_model().parameters())} params")
    print(f"train set: N={TRAIN_N}, batch={TRAIN_BATCH}")
    print(f"THREE DISJOINT probe sets, N={PROBE_N} each: B_stale (-> g_stale), B_ref (-> g_ref, "
          f"the direction-predictor target), B_eval (-> delta_loss, the usefulness TARGET; "
          f"never used to compute the predictor)")
    print(f"total training steps: {TOTAL_STEPS}; stale ages k tested: {K_LIST}")
    print(f"PRIMARY probe step: normalized direction g_dir = g_stale/||g_stale||, "
          f"eta in {ETA_PROBE_NORMALIZED_LIST} (default {ETA_PROBE_NORM_DEFAULT})")
    print(f"CONTROL probe step: raw g_stale (v1-style), eta in {ETA_PROBE_RAW_LIST} "
          f"(default {ETA_PROBE_RAW_DEFAULT})")
    print(f"EMA betas tested: {EMA_BETAS}")
    print("NOTE: staleness here is a controlled proxy (this learner's own past snapshots),")
    print("      not real asynchronous / distributed rollout staleness. See module docstring.")
    print("NOTE (v2 fixes vs. v1): predictor (cosine on B_stale/B_ref) and target (delta_loss on")
    print("      B_eval) now share no batch; the PRIMARY probe update uses a magnitude-normalized")
    print("      stale direction so all stale gradients take the same-size step.")

    print()
    print("=" * 92)
    print("PART 1 (PRIMARY): SGD trajectory")
    print("=" * 92)
    sgd_traj, sgd_obs = run_for_optimizer(
        "SGD", torch.optim.SGD, dict(lr=0.05, momentum=0.9),
        x_train, y_train, x_stale, y_stale, x_ref, y_ref, x_eval, y_eval, SGD_T_STRIDE,
    )

    print()
    print("=" * 92)
    print("PART 2 (SUPPLEMENTARY): AdamW trajectory")
    print("=" * 92)
    adamw_traj, adamw_obs = run_for_optimizer(
        "AdamW", torch.optim.AdamW, dict(lr=1e-2),
        x_train, y_train, x_stale, y_stale, x_ref, y_ref, x_eval, y_eval, ADAMW_T_STRIDE,
    )

    print()
    print("=" * 92)
    print("CORE ANALYSIS 1: age vs. usefulness  (SGD)")
    print("=" * 92)
    age_result = analysis_age_vs_usefulness(sgd_obs)

    print()
    print("=" * 92)
    print("CORE ANALYSIS 2: directional consistency vs. usefulness  (SGD)")
    print("=" * 92)
    print(">>> vs. reference gradient (B_ref, disjoint from B_stale and B_eval)")
    dir_ref_result = analysis_direction_vs_usefulness(sgd_obs, "cosine_stale_ref")
    print()
    for beta in EMA_BETAS:
        print(f">>> vs. EMA(beta={beta}) historical direction")
        analysis_direction_vs_usefulness(sgd_obs, f"cosine_stale_ema_{beta}")
        print()

    print("=" * 92)
    print("CORE ANALYSIS 3: age vs. direction - which predicts usefulness better?  (SGD, PRIMARY label)")
    print("=" * 92)
    predictive_result = analysis_predictive_power(sgd_obs)

    print()
    print("=" * 92)
    print("ANALYSIS: raw (v1-style) vs. normalized (v2 primary) probe update  (SGD)")
    print("=" * 92)
    raw_vs_norm_result = analysis_raw_vs_normalized(sgd_obs)

    print()
    print("=" * 92)
    print("CORE ANALYSIS 4: same-age variance  (SGD)")
    print("=" * 92)
    same_age_result = analysis_same_age_variance(sgd_obs)

    print()
    print("=" * 92)
    print("CORE ANALYSIS 5: old-but-aligned vs. young-but-conflicting  (SGD)")
    print("=" * 92)
    contrast_result = analysis_old_aligned_vs_young_conflicting(sgd_obs)

    print()
    print("=" * 92)
    print("LAYER-WISE extension  (SGD)")
    print("=" * 92)
    layerwise_result = analysis_layerwise(sgd_obs, sgd_traj["names"])

    print()
    print("=" * 92)
    print("EMA direction study (supplementary)  (SGD)")
    print("=" * 92)
    analysis_ema_study(sgd_obs)

    print()
    print("=" * 92)
    print("eta_probe sensitivity check  (SGD)")
    print("=" * 92)
    analysis_eta_sensitivity(sgd_obs)

    print()
    print("=" * 92)
    print("SUPPLEMENTARY: same core analyses on AdamW trajectory")
    print("=" * 92)
    adamw_age_result = analysis_age_vs_usefulness(adamw_obs)
    print()
    adamw_dir_result = analysis_direction_vs_usefulness(adamw_obs, "cosine_stale_ref")
    print()
    adamw_predictive_result = analysis_predictive_power(adamw_obs)

    print()
    print("=" * 92)
    print("LIMITATIONS")
    print("=" * 92)
    print("- Staleness is simulated via one learner's own historical snapshots on fixed probe")
    print("  batches, not real asynchronous rollout staleness (no actor lag, no partial rollouts,")
    print("  no RLVR, no distributed queueing).")
    print("- Tiny CPU-only MLP + synthetic teacher-generated regression data; conclusions may not")
    print("  transfer directly to large models, classification/RL losses, or long-horizon staleness.")
    print("- 'Usefulness' is operationalized as delta_loss on B_eval after a single small probe")
    print("  step; it is one reasonable proxy, not a ground truth.")
    print("- v1's two known confounds are addressed here (see module docstring): predictor and")
    print("  target no longer share a batch/gradient, and the primary probe step uses a")
    print("  magnitude-normalized direction. Residual, weaker couplings remain: B_ref, B_stale and")
    print("  B_eval are each FIXED across all t (not resampled per observation), and all three are")
    print("  drawn from the same teacher-generated distribution (independent samples, not an")
    print("  independent data-generating process).")
    print("- No scipy/sklearn in this environment: Spearman correlation is a hand-rolled")
    print("  average-rank implementation, and 'predictive power' is assessed via threshold sweeps")
    print("  (precision/recall/F1) rather than a fitted classifier (e.g. logistic regression).")
    print("- Pearson correlation assumes a roughly linear relationship; where it disagrees with")
    print("  Spearman, prefer Spearman's rank-based reading.")
    print(f"- Only {len(K_LIST)} discrete ages ({K_LIST}) and one architecture/optimizer-hyperparameter")
    print("  setting per optimizer were tested; no hyperparameter sweep.")
    print("- The normalized probe step uses one global eta regardless of how parameter-space scale")
    print("  differs across layers; a per-layer-normalized step was not tested.")

    print()
    print("=" * 92)
    print("FINAL ANSWERS (from the numbers above, not assumed in advance)")
    print("=" * 92)
    age_p, age_s = age_result["pearson"], age_result["spearman"]
    dir_p, dir_s = dir_ref_result["pearson"], dir_ref_result["spearman"]
    age_f1 = predictive_result["age_best"][1]
    dir_f1 = predictive_result["cos_best"][1]
    combo_f1 = predictive_result["combo_best"][2]

    print("1. After decoupling predictor and target into three disjoint batches, does directional")
    print("   consistency still out-predict age?")
    print(f"   corr(age, delta_loss): pearson={age_p:.4f}, spearman={age_s:.4f}; "
          f"best age-only F1={age_f1:.4f}.")
    print(f"   corr(cosine_stale_ref, delta_loss): pearson={dir_p:.4f}, spearman={dir_s:.4f}; "
          f"best direction-only F1={dir_f1:.4f}; best combined F1={combo_f1:.4f}.")
    if dir_f1 > age_f1 + 0.02:
        print("   -> YES, direction still out-performs age as a usefulness predictor after decoupling.")
    elif dir_f1 < age_f1 - 0.02:
        print("   -> NO, age out-performs direction once predictor and target are decoupled.")
    else:
        print("   -> Roughly comparable once decoupled; check the combined-rule F1 above for any edge.")

    print()
    print("2. After also normalizing the stale gradient (removing the magnitude confound), does")
    print("   the conclusion still hold?")
    print(f"   normalized: age-only F1={raw_vs_norm_result['age_f1_norm']:.4f}, "
          f"direction-only F1={raw_vs_norm_result['cos_f1_norm']:.4f}")
    print(f"   raw control: age-only F1={raw_vs_norm_result['age_f1_raw']:.4f}, "
          f"direction-only F1={raw_vs_norm_result['cos_f1_raw']:.4f}")
    print(f"   sign agreement between normalized-useful and raw-useful labels: "
          f"{raw_vs_norm_result['sign_agree']:.4f}")
    norm_vs_raw_consistent = abs(raw_vs_norm_result["cos_f1_norm"] - raw_vs_norm_result["cos_f1_raw"]) < 0.1
    if norm_vs_raw_consistent:
        print("   -> YES, normalizing does not qualitatively change the direction-vs-age comparison.")
    else:
        print("   -> Normalizing DOES materially change the picture - the raw-step result was partly")
        print("      driven by the magnitude confound; trust the normalized numbers over the raw ones.")

    print()
    print("3. Did v1's near-perfect direction-only F1 (=1.0000) drop once the two confounds were")
    print("   removed?")
    print(f"   v1 (coupled batch, raw magnitude): direction-only F1={V1_DIRECTION_ONLY_BEST_F1:.4f}")
    print(f"   v2 (decoupled batches, normalized step): direction-only F1={dir_f1:.4f}")
    if dir_f1 < V1_DIRECTION_ONLY_BEST_F1 - 0.02:
        print("   -> YES, it dropped, confirming v1's F1=1.0 was inflated by the coupling/magnitude")
        print("      confounds rather than being a clean, generalizable result.")
    elif dir_f1 >= V1_DIRECTION_ONLY_BEST_F1 - 0.02:
        print("   -> NO clear drop: direction-only F1 remained comparably high even after removing")
        print("      both confounds - this is a stronger result than v1's, since it is no longer")
        print("      explainable by shared-batch or magnitude artifacts.")

    print()
    print("4. Does directional consistency still carry information at a FIXED age (same-age check)?")
    print(f"   Same-age cosine/usefulness contrast: "
          f"{'FOUND' if same_age_result['contrast_found'] else 'NOT found'} (see analysis 4's "
          f"within_age_corr column for per-age correlation strength).")
    print(f"   Old-but-aligned examples found: {contrast_result['n_old_aligned']}; "
          f"young-but-conflicting examples found: {contrast_result['n_young_conflicting']}.")

    print()
    print("5. Is it worth proceeding to actually test w(c) * g gradient reweighting?")
    worth_it = (
        (dir_f1 > age_f1 + 0.02)
        and norm_vs_raw_consistent
        and same_age_result["contrast_found"]
    )
    if worth_it:
        print("   -> Tentatively YES: even after removing the batch-coupling and magnitude confounds,")
        print("      directional consistency still shows predictive signal beyond age, including at")
        print("      fixed age. This is a materially more rigorous prerequisite than v1 cleared.")
        print("      Phase 2 (testing w(c) * g) should still start under this same controlled-proxy")
        print("      staleness setup before any claim about real async training.")
    else:
        print("   -> Tentatively NO / NOT YET: once the confounds are removed, the evidence for")
        print("      direction adding information beyond age is weaker or inconsistent (see numbers")
        print("      above). Re-examine before investing in a reweighting algorithm rather than")
        print("      proceeding on the strength of the v1 result alone.")


if __name__ == "__main__":
    main()
