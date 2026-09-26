"""E20 gradient check: analytic vs finite-difference gradients of the exact race network (tiny, no data).

With the near-miss surrogate off (sigma_b -> 0) the analytic gradient is the exact derivative within the
realised piece; small weight perturbations that do not change the firing pattern must agree.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e20_exact import ExactRaceNet  # noqa: E402


def loss_of(net, tin, y):
    _, _, To = net.forward(tin)
    z = -To / net.tau
    z = z - z.max(1, keepdims=True)
    return float(-(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean())


def main():
    rng = np.random.default_rng(0)
    d, b = 30, 16
    tin = np.where(rng.random((b, d)) < 0.5, rng.random((b, d)) * 0.8, np.inf).astype(np.float32)
    y = rng.integers(0, 4, b)
    drive = float(np.where(np.isfinite(tin), 1 - tin, 0).sum(1).mean())
    net = ExactRaceNet(d, [20, 20], 4, drive, rng, sigma_b=1e-9)
    # analytic gradients: capture them by running one step with lr = 0 and reading the Adam inputs
    acts, x, To = net.forward(tin)
    z = -To / net.tau
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    gz = p.copy(); gz[np.arange(b), y] -= 1
    gT = -gz / net.tau / b
    last = acts[-1]
    soft = (np.where(np.isfinite(last["T"]) & (last["presence"] > 0), last["T"], np.inf).astype(np.float32),
            last["presence"])
    gWo, gt = net._node_grads(x, To, net.Wo, gT, soft)
    L1 = acts[1]
    gT1 = np.where(np.isfinite(L1["T"]), gt, 0.0)
    P = acts[0]
    soft0 = (np.where(np.isfinite(P["T"]) & (P["presence"] > 0), P["T"], np.inf).astype(np.float32), P["presence"])
    gW1, _ = net._node_grads(L1["tin"], L1["T"], net.W[1], gT1, soft0)
    worst = 0.0
    for name, W, G in (("Wo", net.Wo, gWo), ("W1", net.W[1], gW1)):
        errs = []
        for _ in range(40):
            i, j = rng.integers(0, W.shape[0]), rng.integers(0, W.shape[1])
            if abs(G[i, j]) < 1e-6:
                continue
            h = 1e-4 * max(abs(W[i, j]), 1e-3)
            old = W[i, j]
            W[i, j] = old + h; lp = loss_of(net, tin, y)
            W[i, j] = old - h; lm = loss_of(net, tin, y)
            W[i, j] = old
            fd = (lp - lm) / (2 * h)
            errs.append(abs(fd - G[i, j]) / max(abs(fd), abs(G[i, j]), 1e-8))
        med = float(np.median(errs)) if errs else float("nan")
        worst = max(worst, med)
        print(f"{name}: {len(errs)} coordinates, median relative error {med:.2e}", flush=True)
    print("GRADCHECK", "PASS" if worst < 1e-2 else "FAIL", flush=True)


if __name__ == "__main__":
    main()
