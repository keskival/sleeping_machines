"""Replay actual shared-router parameter changes, including all race changes.

The update is chosen on four fitting utterances; evaluate its finite loss
change on that batch and four held-out-speaker utterances. This is a small
local diagnostic, never a checkpoint update or a benchmark claim.
"""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from e117_serial_event_shd import batch, load_items
from e118_race_carrier_shd import RaceNet, OUT


def evaluate(net, inputs):
    net.eval()
    with torch.no_grad():
        logits, _, _, trace = net(*inputs[:4], len(inputs[-1]), trace=True)
        loss = float(F.cross_entropy(logits, inputs[-1]))
        winners = [t["winner"].clone() for t in trace]
    return loss, winners


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    output = OUT/(a.tag+'.json')
    if output.exists():
        raise FileExistsError(output)
    saved = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    s = saved["args"]
    net = RaceNet(depth=s["depth"], cf_credit=True)
    net.load_state_dict(saved["state_dict"])
    fit = load_items(40,.01,s["limit"],"fit_spk",s["seed"])
    dev = load_items(40,.01,s["eval_limit"],"val_spk",s["seed"]+1)
    adapt, transfer = batch(fit[:4]), batch(dev[:4])
    params = [p for layer in net.layers for p in (layer.route, layer.route_bias)]
    original = [p.detach().clone() for p in params]
    base_fit, fit_winners = evaluate(net, adapt)
    base_dev, dev_winners = evaluate(net, transfer)
    rows = []
    for credit in (False, True):
        for layer in net.layers:
            layer.cf_credit = credit
        net.train()
        logits = net(*adapt[:4], len(adapt[-1]))[0]
        loss = F.cross_entropy(logits, adapt[-1])
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        gradients = [torch.zeros_like(p) if g is None else g for p,g in zip(params, gradients)]
        norm = torch.sqrt(sum(g.square().sum() for g in gradients))
        for step in (.001, .01, .1):
            with torch.no_grad():
                for p, root, g in zip(params,original,gradients):
                    p.copy_(root-step*g/norm.clamp_min(1e-12))
            lf, wf = evaluate(net, adapt)
            ld, wd = evaluate(net, transfer)
            rows.append({"counterfactual_credit": credit,"router_step_l2": step,
                         "router_gradient_l2": float(norm),
                         "predicted_adapt_loss_change": -step*float(norm),
                         "actual_adapt_loss_change": lf-base_fit,
                         "actual_development_loss_change": ld-base_dev,
                         "adapt_winner_changes": [int((x!=y).sum()) for x,y in zip(wf,fit_winners)],
                         "development_winner_changes": [int((x!=y).sum()) for x,y in zip(wd,dev_winners)]})
            with torch.no_grad():
                for p, root in zip(params,original):
                    p.copy_(root)
    result = {"checkpoint":a.checkpoint,"protocol":"4 fitting to choose direction, 4 held-out to evaluate transfer; no saved model mutation",
              "base_adapt_nll":base_fit,"base_development_nll":base_dev,
              "fit_ids":[x[4] for x in fit[:4]],"dev_ids":[x[4] for x in dev[:4]],"rows":rows}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
