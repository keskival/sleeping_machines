"""Exact finite examples for THEORY §§159–163; no SHD performance claim."""
import json
from pathlib import Path

import numpy as np


def option_values(payoffs):
    q = np.asarray(payoffs, dtype=float)  # equiprobable observed scenarios × actions
    adaptive = float(q.max(axis=1).mean())
    committed = float(q.mean(axis=0).max())
    return {"adaptive": adaptive, "committed": committed,
            "premium": adaptive - committed,
            "without_observing_scenario": committed}


def gradient_values(gradients):
    g = np.asarray(gradients, dtype=float)
    return {"same_sample": float((g * g).sum(1).mean()),
            "independent_sample_transfer": float((g @ g.T).mean()),
            "squared_mean": float(g.mean(0) @ g.mean(0)),
            "variance_trace": float(g.var(axis=0).sum())}


def main():
    result = {
        "scope": "Exact toy calculations, identity update metric; not learned-model results.",
        "complementary_routes": option_values([[1, -1], [-1, 1]]),
        "one_volatile_route": option_values([[1], [-1]]),
        "duplicate_routes": option_values([[1, 1], [-1, -1]]),
        "zero_mean_noisy_gradient": gradient_values([[2], [-2]]),
        "consistent_gradient": gradient_values([[1], [1]]),
        "premium_backup_counterexample": {"child_premiums": [0, 9],
                                            "root_premium": max(10, 9) - max(10, 0)},
        "incompatible_shared_control": {
            "margin_1": "xi - 1", "margin_2": "-xi - 1",
            "sum_of_margins": -2, "both_nonnegative_feasible": False,
            "reason": "The two margins sum to -2 for every xi."},
        "bernoulli_half_gain_curve": [{"draws": k, "expected_best_gain": 1 - 0.5 ** k,
                                        "next_draw_marginal_gain": 0.5 ** (k + 1)}
                                       for k in range(1, 6)],
    }
    # Analytical contracts; no Monte Carlo tolerances or favorable seed selection.
    assert result["complementary_routes"]["premium"] == 1
    assert result["one_volatile_route"]["premium"] == 0
    assert result["duplicate_routes"]["premium"] == 0
    for key in ("zero_mean_noisy_gradient", "consistent_gradient"):
        v = result[key]
        assert v["same_sample"] == v["squared_mean"] + v["variance_trace"]
        assert v["independent_sample_transfer"] == v["squared_mean"]
    out = Path(__file__).parent / "results/e116/optionality_contract_v2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite completed result: {out}")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
