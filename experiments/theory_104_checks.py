"""Numerical checks of THEORY §104: (a) compensator estimator E[Lambda_i(T)] = P(i wins) for time-varying, non-proportional
hazards; (b) proportional hazards: winner independent of T, P = softmax for any common gain; (c) races with drifting logits
are continuous mixtures of softmaxes whose log-probability matrix exceeds the single-softmax rank bound d+1;
(d) time-change gradient dT/dtheta = -dLambda(T)/dtheta / lambda(T) vs finite differences."""
import json, os
import numpy as np

rng = np.random.default_rng(0); out = {}
# ---- (a) non-proportional hazards lambda_i(t) = exp(s_i + c_i t); first arrival of each: Lambda_i(T_i) = E_i
K = 5; s = rng.normal(0, 1, K); c = rng.normal(0, 1.5, K)
def Lam(t, s, c):
    return np.where(np.abs(c) < 1e-9, np.exp(s) * t, np.exp(s) * np.expm1(c * t) / c)
def first(E, s, c):
    arg = 1 + c * E * np.exp(-s)
    return np.where(arg > 0, np.log(np.maximum(arg, 1e-300)) / c, np.inf)
tg = np.linspace(0, 40, 400001); lam = np.exp(s[:, None] + c[:, None] * tg); L = np.cumsum(lam, 1) * (tg[1] - tg[0])
dens = lam * np.exp(-L.sum(0)); P_exact = dens.sum(1) * (tg[1] - tg[0])
n = 400000; E = rng.exponential(size=(n, K)); Ti = first(E, s, c); w = Ti.argmin(1); T = Ti.min(1)
ind = np.bincount(w, minlength=K) / n
comp = Lam(T[:, None], s, c)                                          # each unit's own integrated hazard at T
out["a_compensator"] = {"P_exact": P_exact.round(4).tolist(), "winner_freq": ind.round(4).tolist(),
                        "mean_Lambda_i(T)": comp.mean(0).round(4).tolist(),
                        "var_indicator": (ind * (1 - ind)).round(4).tolist(), "var_Lambda": comp.var(0).round(4).tolist(),
                        "sum_Lambda_mean_var": [float(comp.sum(1).mean().round(4)), float(comp.sum(1).var().round(4))]}
# ---- (b) proportional hazards: lambda_i(t) = exp(s_i) g(t), g(t) = 1 + sin(3t)^2 * 4 (arbitrary common gain)
G = lambda t: 3 * t - np.sin(6 * t) / 3                                 # integral of g = 1 + 4 sin(3t)^2 = 3 - 2cos(6t)
Ginv_grid = np.linspace(0, 60, 600001); Gv = G(Ginv_grid)
E = rng.exponential(size=(n, K)); Ti = np.interp(E * np.exp(-s), Gv, Ginv_grid); w = Ti.argmin(1); T = Ti.min(1)
sm = np.exp(s) / np.exp(s).sum()
fast = T < np.median(T)
out["b_proportional"] = {"softmax": sm.round(4).tolist(), "winner_freq": (np.bincount(w, minlength=K) / n).round(4).tolist(),
                         "winner_freq_fast_half": (np.bincount(w[fast], minlength=K) / fast.sum()).round(4).tolist(),
                         "winner_freq_slow_half": (np.bincount(w[~fast], minlength=K) / (~fast).sum()).round(4).tolist(),
                         "mean_exp(s_i)G(T)": (np.exp(s)[None] * G(T)[:, None]).mean(0).round(4).tolist()}
# ---- (c) mixture-of-softmaxes rank: logits drift during the race, s_i(t) = w_i . (h0 + t u) - m, contexts h0 in R^d;
# the common offset m slows the race (proportional gain e^-m) and so sets how much of the logit trajectory it integrates
d, V, N = 4, 60, 300; Wm = rng.normal(0, 1, (V, d)); U = rng.normal(0, 1, (N, d)); H0 = rng.normal(0, 1, (N, d))
out["c_mos_rank"] = {"d": d, "single_softmax_rank_bound": d + 1, "sweep": []}
for m in (0, 3, 6, 9):
    tt = np.linspace(0, 12 * np.exp(m / 2), 8001); dt = tt[1] - tt[0]; Lg = np.empty((N, V)); ET = []
    for k in range(N):
        S = np.minimum((H0[k][None] + tt[:, None] * U[k][None]) @ Wm.T - m, 50)
        lam = np.exp(S); f = lam * np.exp(-np.cumsum(lam.sum(1)) * dt)[:, None]
        P = f.sum(0) * dt; ET.append(float((f.sum(1) * tt).sum() * dt / P.sum())); Lg[k] = np.log(np.maximum(P / P.sum(), 1e-300))
    sv = np.linalg.svd(Lg - Lg.mean(1, keepdims=True), compute_uv=False)
    out["c_mos_rank"]["sweep"].append({"slowdown_log": m, "mean_decision_time": round(float(np.mean(ET)), 4),
                                       "sv_5_to_8": sv[4:8].round(3).tolist(),
                                       "residual_beyond_rank_d+1": round(float(np.sqrt((sv[d + 1:] ** 2).sum() / (sv ** 2).sum())), 4)})
# ---- (d) time-change gradient for lambda(t) = exp(a + b t), d T / d a and d T / d b with common random numbers
a, b, e = 0.3, -0.4, 0.7
Tf = lambda a, b: float(first(np.array(e), a, b))
T0 = Tf(a, b); lamT = np.exp(a + b * T0)
dLa = Lam(T0, a, b); dLb = np.exp(a) * (T0 * np.exp(b * T0) / b - np.expm1(b * T0) / b ** 2)
h = 1e-6
out["d_time_change_grad"] = {"dT/da_ift": -dLa / lamT, "dT/da_fd": (Tf(a + h, b) - Tf(a - h, b)) / (2 * h),
                             "dT/db_ift": -dLb / lamT, "dT/db_fd": (Tf(a, b + h) - Tf(a, b - h)) / (2 * h)}
os.makedirs("results/theory", exist_ok=True)
json.dump(out, open("results/theory/s104_checks.json", "w"), indent=1); print(json.dumps(out, indent=1))
