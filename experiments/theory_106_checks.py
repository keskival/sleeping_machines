"""Numerical checks of THEORY §106. (a) Delay-attention work law: keys that must send for softmax mass 1 - eps over N
Gaussian scores of spread sigma, against the tilted-Gaussian prediction N Q(sigma - z_eps) and the frozen (REM) phase above
sigma_c = sqrt(2 ln N) where O(1) keys suffice. (b) Dilation covariance: a time-vector unit whose mode rates, content delays
and reset rate are scaled by rho^m, driven by the input stream dilated by rho^m, fires at exactly rho^m times the original
firing times."""
import json
import math
import os

import numpy as np
from statistics import NormalDist

rng = np.random.default_rng(0); out = {"a_work_law": [], "b_dilation": {}}
eps = 0.01; ND = NormalDist(); z = ND.inv_cdf(1 - eps)
for N in (10_000, 100_000, 1_000_000):
    sc = math.sqrt(2 * math.log(N))
    for sig in (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        need = []
        for rep in range(5 if N < 1_000_000 else 2):
            s = rng.normal(0, sig, N); p = np.exp(s - s.max()); p /= p.sum(); ps = np.sort(p)[::-1]
            need.append(int(np.searchsorted(np.cumsum(ps), 1 - eps)) + 1)
        pred = N * (1 - ND.cdf(sig - z))
        out["a_work_law"].append({"N": N, "sigma": sig, "sigma_c": round(sc, 2), "keys_needed_median": int(np.median(need)),
                                  "tilted_prediction": round(float(pred), 1), "fraction": float(np.median(need)) / N})
# (b) dilation covariance
n, d = 4, 3
lam0 = np.array([-0.05 + 0.3j, -0.1 + 0.0j, -0.02 + 0.1j, -0.2 + 0.7j]); B = rng.normal(0, 1, (n, d)) + 1j * rng.normal(0, 1, (n, d))
w = rng.normal(0, 0.4, n) + 1j * rng.normal(0, 0.4, n); q = rng.normal(0, 1, d); c = 0.3; tau0 = 5.0; tauR = 20.0; th = 1.0
ts = np.sort(rng.uniform(0, 200, 60)); vs = rng.normal(0, 1, (60, d))
def fire_times(ts, vs, a):
    lam = lam0 / a; tau = tau0 * a; tR = tauR * a                       # a unit at scale a: rates / a, delays and reset * a
    r = vs @ q + c; send = r > 0
    arr = ts[send] + tau * r[send]; bv = vs[send] @ B.T
    grid = np.arange(0, 400 * a, 0.002 * a); T = []; Rs = []
    V = np.zeros(len(grid))
    for k, t in enumerate(grid):                                      # potential on a fine grid, reset by subtraction
        m = arr <= t
        zt = (np.exp(np.outer(t - arr[m], lam)) * bv[m]).sum(0)
        R = sum(math.exp(-(t - T0) / tR) for T0 in T)
        V[k] = (w * zt).real.sum() - th * R
        if V[k] >= th:
            T.append(t)
    return np.array(T)
T1 = fire_times(ts, vs, 1.0); a = 1.7; T2 = fire_times(ts * a, vs, a)
out["b_dilation"] = {"spikes_original": len(T1), "spikes_dilated": len(T2), "dilation": a,
                     "max_abs_diff_T_dilated_vs_a_T": float(np.max(np.abs(T2 - a * T1))) if len(T1) == len(T2) else None,
                     "grid_step_dilated": 0.002 * a}
os.makedirs("results/theory", exist_ok=True)
json.dump(out, open("results/theory/s106_checks.json", "w"), indent=1)
for r in out["a_work_law"]:
    print(r)
print(out["b_dilation"])
