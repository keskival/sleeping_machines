"""E76: the attention work law of THEORY §106(a), measured in trained character-level Transformers (E64b checkpoints).

For every layer, head and query position t (t + 1 visible keys, causal), the number of keys that must send for softmax mass
1 - eps (the cost of delay-coded attention, §105(b)), the spread sigma of the query's scores, and the tilted-Gaussian
prediction (t + 1) Q(sigma - z_eps). Aggregated by context size to give the work exponent: keys needed ~ N^alpha.
"""
import argparse
import glob
import json
import math
import os
import sys
from statistics import NormalDist

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import e62_charlm as S1  # noqa: E402
import e64_lm_baselines as B64  # noqa: E402

torch.set_num_threads(1)
OUT = os.path.join(os.path.dirname(__file__), "results", "e76")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--eps", default="0.01,0.05")
    ap.add_argument("--checkpoint_dir", default=os.path.join(os.path.dirname(__file__), "results", "e64"),
                    help="Directory containing the completed E64 Transformer checkpoints")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    x = S1.load(); valid = torch.tensor(x[90_000_000:90_000_000 + 1_000_000])
    ND = NormalDist(); res = {}
    checkpoints = sorted(glob.glob(os.path.join(a.checkpoint_dir, "tf_*_v.pt")))
    if not checkpoints:
        raise FileNotFoundError(f"No completed Transformer checkpoints in {a.checkpoint_dir}")
    for ck in checkpoints:
        c = torch.load(ck); ar = c["args"]
        net = B64.TfLM(ar["size"], ar["layers"], ar["ctx"]); net.load_state_dict(c["state"]); net.eval()
        T = ar["ctx"]; H = 4; dh = ar["size"] // H
        caught = []
        hooks = [l.self_attn.register_forward_pre_hook(lambda m, args, kw=None: caught.append(args[0].detach()))
                 for l in net.enc.layers]
        rows = {e: [] for e in a.eps.split(",")}; sig_rows = []
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for s0 in rng.integers(0, len(valid) - T - 1, a.windows):
                caught.clear(); net(valid[s0:s0 + T][None])
                for li, (h, layer) in enumerate(zip(caught, net.enc.layers)):
                    W, bias = layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias
                    q = (h[0] @ W[:ar["size"]].T + bias[:ar["size"]]).view(T, H, dh).transpose(0, 1)
                    k = (h[0] @ W[ar["size"]:2 * ar["size"]].T + bias[ar["size"]:2 * ar["size"]]).view(T, H, dh).transpose(0, 1)
                    sc = q @ k.transpose(1, 2) / math.sqrt(dh)                               # (H, T, T)
                    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
                    sc = sc.masked_fill(mask, float("-inf")); p = torch.softmax(sc, -1)
                    ps, _ = torch.sort(p, -1, descending=True); cum = torch.cumsum(ps, -1)
                    nkeys = torch.arange(1, T + 1)[None, :].expand(H, T)                     # visible keys per query
                    scm = sc.masked_fill(mask, float("nan"))
                    sig = torch.from_numpy(np.nanstd(scm.numpy(), -1))                       # spread of the query's scores
                    for e in rows:
                        need = (cum < 1 - float(e)).sum(-1) + 1
                        rows[e].append(torch.stack([torch.full_like(need, li), torch.arange(H)[:, None].expand(H, T),
                                                    nkeys, need], -1).reshape(-1, 4))
                    sig_rows.append(torch.stack([torch.full((H, T), li), nkeys, sig], -1).reshape(-1, 3))
        for hk in hooks:
            hk.remove()
        out = {"ckpt": os.path.basename(ck), "args": ar, "by_eps": {}}
        sig_all = torch.cat(sig_rows)
        for e, rr in rows.items():
            R = torch.cat(rr).numpy(); z = ND.inv_cdf(1 - float(e)); table = []
            for lo, hi in ((8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 257)):
                m = (R[:, 2] >= lo) & (R[:, 2] < hi)
                if not m.any():
                    continue
                sg = sig_all.numpy()[m, 2]; N = R[m, 2]
                pred = N * np.array([1 - ND.cdf(float(s_) - z) for s_ in sg])
                table.append({"keys_visible": [lo, hi - 1], "median_needed": float(np.median(R[m, 3])),
                              "mean_needed": float(R[m, 3].mean()), "median_fraction": float(np.median(R[m, 3] / N)),
                              "median_sigma": float(np.median(sg)), "median_tilted_pred": float(np.median(pred)),
                              "per_layer_median": [float(np.median(R[m & (R[:, 0] == l), 3])) for l in range(ar["layers"])]})
            xs = np.log([np.mean(r["keys_visible"]) for r in table]); ys = np.log([r["mean_needed"] for r in table])
            out["by_eps"][e] = {"table": table, "work_exponent_alpha": float(np.polyfit(xs, ys, 1)[0])}
        res[os.path.basename(ck)] = out
        print(json.dumps({"ckpt": out["ckpt"], **{e: v["work_exponent_alpha"] for e, v in out["by_eps"].items()}}), flush=True)
    json.dump(res, open(os.path.join(OUT, "attention_work.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
