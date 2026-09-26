"""M39 (THEORY §37): how the information in an input patch grows with patch size.

Hierarchical layer sizing needs the scaling exponent α in h(s) ∝ s^α, where h(s) is the
information carried by an s×s patch of the latency code. Estimated with the Gaussian channel
bound h = ½ Σ log2(1 + λ_i/ν) over the patch covariance's eigenvalues λ_i, at a timing-noise floor
ν (the code's resolution), averaged over patch positions. Cheap: covariances of at most 784×784.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e6_hidden import HORIZON, latency_code, mnist  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results", "theory")


def patch_bits(tt, s, nu, stride):
    img = np.where(np.isfinite(tt), tt, HORIZON).reshape(-1, 28, 28).astype(np.float64)
    bits = []
    for r in range(0, 28 - s + 1, stride):
        for c in range(0, 28 - s + 1, stride):
            p = img[:, r:r + s, c:c + s].reshape(len(img), -1)
            lam = np.clip(np.linalg.eigvalsh(np.cov(p, rowvar=False).reshape(s * s, s * s)), 0, None)
            bits.append(0.5 * np.log2(1 + lam / nu).sum())
    return float(np.mean(bits))


def main():
    x, _ = mnist("train")
    tt = latency_code(x[:10000])
    res = {}
    for nu in (0.01 ** 2, 0.03 ** 2, 0.1 ** 2):
        sizes = [1, 2, 4, 7, 14, 28]
        h = [patch_bits(tt, s, nu, stride=max(s // 2, 1) if s < 28 else 1) for s in sizes]
        area = np.array(sizes, float) ** 2
        slope = np.polyfit(np.log(area[1:]), np.log(np.array(h[1:])), 1)[0]   # h ∝ area^α
        res[f"nu={nu:g}"] = {"sizes": sizes, "bits": h, "alpha_area": float(slope)}
        print(f"noise sd {np.sqrt(nu):.2f}: bits {np.round(h, 1).tolist()}  alpha(area) = {slope:.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "m39_patch_entropy.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
