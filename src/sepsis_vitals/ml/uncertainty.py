"""Paired, patient-clustered bootstrap confidence intervals for model comparisons.

Estimand
--------
For each model (arm) evaluated on the same held-out rows: the row-level AUROC
and AUPRC over the held-out population, and the paired difference between
two arms (arm B minus arm A). These are the quantities the ablation and
profile reports already state as point estimates.

Resampling
----------
Rows from one patient are correlated (consecutive observations), so the
resampling unit is the **patient**. Each replicate draws patients with
replacement and keeps every row of each drawn patient. Both arms are scored
on the same replicate (paired), so the interval for the difference reflects
the correlation between the arms. Intervals are percentile intervals at the
stated level, with a fixed seed.

A replicate with only one class cannot define AUROC or AUPRC. It is counted
as degenerate and excluded, and the report states how many replicates were
excluded. When more than 10% are degenerate, the interval is marked
unreliable.

What the intervals do and do not cover
--------------------------------------
They describe sampling variability of the held-out set *given the trained
models*. They do not include training variability, and on synthetic or
in-sample data they are not evidence of performance in patients.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def _auroc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def _auprc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


METRICS: Dict[str, Metric] = {"auroc": _auroc, "auprc": _auprc}


def _interval(values: Sequence[float], level: float) -> Optional[list]:
    if not len(values):
        return None
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(np.asarray(values), [alpha, 1.0 - alpha])
    return [round(float(lo), 4), round(float(hi), 4)]


def paired_cluster_bootstrap(
    groups: Sequence[Any],
    y_true: Sequence[int],
    predictions: Dict[str, Sequence[float]],
    *,
    reference: Optional[str] = None,
    metrics: Sequence[str] = ("auroc", "auprc"),
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> Dict[str, Any]:
    """Point estimates and percentile CIs for each arm, and paired differences.

    *predictions* maps arm name to scores aligned with *y_true* and *groups*
    (one entry per held-out row). Differences are reported for every arm
    against *reference* (default: the first arm).
    """
    y = np.asarray(y_true).astype(int)
    group_arr = np.asarray(groups)
    arms = {name: np.asarray(p, dtype=float) for name, p in predictions.items()}
    if not arms:
        raise ValueError("at least one arm is required")
    for name, p in arms.items():
        if p.shape != y.shape:
            raise ValueError(f"arm '{name}' has {p.shape[0]} predictions for {y.shape[0]} rows")
        if not np.all(np.isfinite(p)):
            raise ValueError(f"arm '{name}' has non-finite predictions")
    if group_arr.shape != y.shape:
        raise ValueError("groups must have one entry per row")
    if len(np.unique(y)) < 2:
        raise ValueError("the evaluation set has a single class; AUROC/AUPRC are undefined")
    reference = reference or next(iter(arms))
    if reference not in arms:
        raise ValueError(f"unknown reference arm '{reference}'")
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")

    unique_groups, inverse = np.unique(group_arr, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == g) for g in range(len(unique_groups))]

    point = {name: {m: round(METRICS[m](y, p), 4) for m in metrics} for name, p in arms.items()}
    samples: Dict[str, Dict[str, list]] = {name: {m: [] for m in metrics} for name in arms}
    diffs: Dict[str, Dict[str, list]] = {name: {m: [] for m in metrics} for name in arms if name != reference}

    rng = np.random.default_rng(seed)
    degenerate = 0
    for _ in range(n_boot):
        drawn = rng.integers(0, len(rows_of), size=len(rows_of))
        idx = np.concatenate([rows_of[g] for g in drawn])
        y_b = y[idx]
        if y_b.min() == y_b.max():
            degenerate += 1
            continue
        values = {name: {m: METRICS[m](y_b, p[idx]) for m in metrics} for name, p in arms.items()}
        for name in arms:
            for m in metrics:
                samples[name][m].append(values[name][m])
                if name != reference:
                    diffs[name][m].append(values[name][m] - values[reference][m])

    valid = n_boot - degenerate
    result: Dict[str, Any] = {
        "method": "paired patient-level (cluster) bootstrap, percentile intervals",
        "level": level,
        "n_boot": n_boot,
        "valid_replicates": valid,
        "degenerate_replicates": degenerate,
        "reliable": valid >= 0.9 * n_boot,
        "seed": seed,
        "n_patients": int(len(unique_groups)),
        "n_rows": int(len(y)),
        "n_positive_rows": int(y.sum()),
        "n_patients_with_positive_rows": int(len(np.unique(group_arr[y == 1]))),
        "reference_arm": reference,
        "arms": {
            name: {m: {"estimate": point[name][m], "ci": _interval(samples[name][m], level)} for m in metrics}
            for name in arms
        },
        "differences_vs_reference": {
            name: {
                m: {
                    "estimate": round(point[name][m] - point[reference][m], 4),
                    "ci": _interval(diffs[name][m], level),
                }
                for m in metrics
            }
            for name in diffs
        },
    }
    return result
