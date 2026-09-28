#!/usr/bin/env python3
"""E114: finite-difference check of the fixed-support full-sequence bound in THEORY §114.

This is a small synthetic diagnostic, not a model benchmark. It compares the
dense and support-conditioned softmax-attention Jacobians with respect to the
entire input sequence, and checks the row/column block-norm certificate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SEED_COUNT = 3
BETA_VALUES = (0.5, 1.0, 2.0)
LENGTHS = (3, 5, 8)
MODEL_DIM = 4
QK_DIM = 3
VALUE_DIM = 3
OUTPUT_DIM = 4
FD_STEP = 1e-5
TOL = 1e-8
RATIO_BOUND_FLOOR = 1e-8
OUTPUT = Path("experiments/results/e114/sequence_jacobian.json")


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def project_and_attend(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    beta: float,
    retained: np.ndarray | None,
) -> np.ndarray:
    q, k, v = x @ wq, x @ wk, x @ wv
    p = softmax(beta * (q @ k.T))
    if retained is not None:
        masked = p * retained
        p = masked / masked.sum(axis=1, keepdims=True)
    return (p @ v @ wo).reshape(-1)


def finite_difference_jacobian(fun, x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    base_shape = fun(x.reshape(x.shape))
    jac = np.empty((base_shape.size, flat.size), dtype=np.float64)
    for col in range(flat.size):
        plus, minus = flat.copy(), flat.copy()
        plus[col] += FD_STEP
        minus[col] -= FD_STEP
        yp = fun(plus.reshape(x.shape))
        ym = fun(minus.reshape(x.shape))
        jac[:, col] = (yp - ym) / (2.0 * FD_STEP)
    return jac


def spectral_norm(matrix: np.ndarray) -> float:
    if not matrix.size:
        return 0.0
    return float(np.linalg.norm(matrix, ord=2))


def one_case(seed: int, length: int, mode: str, keep: int, beta: float) -> dict:
    rng = np.random.default_rng(seed + 1009 * length + 97 * keep + int(beta * 10))
    x = rng.normal(0.0, 0.7, size=(length, MODEL_DIM))
    if mode == "reused-key":
        x[:] = x[0] + rng.normal(0.0, 0.005, size=x.shape)

    wq = rng.normal(0.0, 0.45, size=(MODEL_DIM, QK_DIM))
    wk = rng.normal(0.0, 0.45, size=(MODEL_DIM, QK_DIM))
    wv = rng.normal(0.0, 0.45, size=(MODEL_DIM, VALUE_DIM))
    wo = rng.normal(0.0, 0.45, size=(VALUE_DIM, OUTPUT_DIM))
    q, k, v = x @ wq, x @ wk, x @ wv
    p = softmax(beta * (q @ k.T))

    retained = np.zeros_like(p, dtype=bool)
    for i in range(length):
        retained[i, np.argsort(p[i])[-keep:]] = True
    mass = (p * retained).sum(axis=1)
    eps = np.maximum(0.0, 1.0 - mass)
    p_sparse = (p * retained) / mass[:, None]
    y = p @ v
    y_sparse = p_sparse @ v

    # Take the largest Euclidean pair distance explicitly.
    d_k = float(np.max(np.linalg.norm(k[:, None, :] - k[None, :, :], axis=-1)))
    d_v = float(np.max(np.linalg.norm(v[:, None, :] - v[None, :, :], axis=-1)))

    norm_wq = spectral_norm(wq)
    norm_wk = spectral_norm(wk)
    norm_wv = spectral_norm(wv)
    norm_wo = spectral_norm(wo)
    c_q = beta * eps * (1.5 - eps) * d_k * d_v
    b = np.zeros((length, length), dtype=np.float64)
    for i in range(length):
        for j in range(length):
            query_term = c_q[i] * norm_wq if i == j else 0.0
            value_term = abs(p[i, j] - p_sparse[i, j]) * norm_wv
            key_term = (
                beta * d_v * np.linalg.norm(q[i]) * norm_wk
                * (abs(p[i, j] - p_sparse[i, j]) + eps[i] * p_sparse[i, j])
            )
            b[i, j] = norm_wo * (query_term + value_term + key_term)

    row_sum = float(b.sum(axis=1).max())
    col_sum = float(b.sum(axis=0).max())
    jac_bound = float(np.sqrt(row_sum * col_sum))
    out_error = float(np.linalg.norm((y - y_sparse) @ wo))
    out_bound = float(norm_wo * d_v * np.linalg.norm(eps))

    dense_fun = lambda z: project_and_attend(z, wq, wk, wv, wo, beta, None)
    sparse_fun = lambda z: project_and_attend(z, wq, wk, wv, wo, beta, retained)
    jac_dense = finite_difference_jacobian(dense_fun, x)
    jac_sparse = finite_difference_jacobian(sparse_fun, x)
    jac_error = spectral_norm(jac_dense - jac_sparse)
    slack = jac_bound - jac_error
    # Probability differences per stored key, the fan-out statistic F_j in §114.
    fanout_f = np.sum(
        np.linalg.norm(q, axis=1)[:, None]
        * (np.abs(p - p_sparse) + eps[:, None] * p_sparse), axis=0
    )

    return {
        "seed": seed,
        "length": length,
        "mode": mode,
        "keep": keep,
        "beta": beta,
        "max_omitted_mass": float(eps.max()),
        "row_sum_R": row_sum,
        "column_sum_C": col_sum,
        "worst_key_F": float(fanout_f.max()),
        "jacobian_error_2norm": jac_error,
        "jacobian_bound_sqrt_RC": jac_bound,
        "jacobian_slack": slack,
        "forward_error_2norm": out_error,
        "forward_bound": out_bound,
        "pass": slack >= -TOL and out_error <= out_bound + TOL,
    }


def main() -> None:
    rows = []
    for length in LENGTHS:
        keep_values = sorted({1, max(1, length // 2), length})
        for mode in ("diffuse", "reused-key"):
            for keep in keep_values:
                for beta in BETA_VALUES:
                    for seed in range(SEED_COUNT):
                        rows.append(one_case(seed, length, mode, keep, beta))

    failed = [r for r in rows if not r["pass"]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    jacobian_ratio_rows = [r for r in rows if r["jacobian_bound_sqrt_RC"] >= RATIO_BOUND_FLOOR]
    forward_ratio_rows = [r for r in rows if r["forward_bound"] >= RATIO_BOUND_FLOOR]
    result = {
        "experiment": "E114",
        "claim": "fixed-support dense-versus-conditioned attention sequence Jacobian is bounded by sqrt(R*C)",
        "method": "central finite differences over full input sequence; spectral norm of Jacobian difference",
        "cases": len(rows),
        "passed": len(rows) - len(failed),
        "failed": len(failed),
        "max_jacobian_error": max(r["jacobian_error_2norm"] for r in rows),
        "max_jacobian_positive_violation": max(
            0.0, max(r["jacobian_error_2norm"] - r["jacobian_bound_sqrt_RC"] for r in rows)
        ),
        "max_forward_positive_violation": max(
            0.0, max(r["forward_error_2norm"] - r["forward_bound"] for r in rows)
        ),
        "relative_ratio_bound_floor": RATIO_BOUND_FLOOR,
        "jacobian_cases_above_ratio_floor": len(jacobian_ratio_rows),
        "max_jacobian_ratio_above_floor": max(
            r["jacobian_error_2norm"] / r["jacobian_bound_sqrt_RC"] for r in jacobian_ratio_rows
        ),
        "forward_cases_above_ratio_floor": len(forward_ratio_rows),
        "max_forward_ratio_above_floor": max(
            r["forward_error_2norm"] / r["forward_bound"] for r in forward_ratio_rows
        ),
        "ratio_note": "Relative ratios omit near-zero bounds; the 1e-8 floor avoids treating finite-difference residue over an exact-zero bound as a meaningful ratio.",
        "failed_cases": failed,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "failed_cases")}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
