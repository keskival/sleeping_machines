"""Fixed-ridge probes of each E117 carrier depth; development data never fits."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from e117_serial_event_shd import SerialEventNet, load_items, batch, OUT


@torch.no_grad()
def features(net, items, bs):
    rows = [[] for _ in range(len(net.layers)+1)]
    for start in range(0, len(items), bs):
        inputs = batch(items[start:start+bs])
        _, _, h = net(*inputs[:4], len(inputs[-1]), return_features=True)
        for r, v in zip(rows, h):
            r.append(v.numpy())
    return [np.concatenate(r).astype(np.float64) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    output = OUT / (a.tag+'.json')
    if output.exists():
        raise FileExistsError(output)
    state = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    s = state["args"]
    net = SerialEventNet(s["bands"], s["dim"], s["depth"], s["groups"], s["beta"])
    net.load_state_dict(state["state_dict"])
    fit = load_items(s["bands"], s["window"], s["limit"], "fit_spk", s["seed"])
    dev = load_items(s["bands"], s["window"], s["eval_limit"], "val_spk", s["seed"]+1)
    hf, hd = features(net, fit, s["bs"]), features(net, dev, s["bs"])
    yf, yd = np.array([x[3] for x in fit]), np.array([x[3] for x in dev])
    target = np.eye(20)[yf]
    prior = target.mean(0)
    rows = []
    for depth, (x, v) in enumerate(zip(hf, hd)):
        mean = x.mean(0)
        scale = np.maximum(x.std(0), 1e-4)
        z, q = (x-mean)/scale, (v-mean)/scale
        cov = z.T@z/len(z)
        cross = z.T@(target-prior)/len(z)
        weights = np.linalg.solve(cov + .01*np.eye(cov.shape[0]), cross)
        pf, pd = (z@weights+prior).argmax(1), (q@weights+prior).argmax(1)
        eig = np.linalg.eigvalsh(cov)
        rows.append({"depth": depth, "fit_correct": int((pf==yf).sum()),
                     "dev_correct": int((pd==yd).sum()), "fit_n": len(yf), "dev_n": len(yd),
                     "median_feature_std": float(np.median(scale)),
                     "covariance_eigenvalues": eig.tolist(),
                     "effective_rank": float(eig.sum()**2 / (eig@eig)),
                     "dev_predictions": pd.tolist()})
    result = {"checkpoint": a.checkpoint, "ridge": .01,
              "protocol": "Frozen features; center/scale/ridge fit on fitting speakers only; all depths reported",
              "rows": rows}
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
