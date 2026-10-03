"""Routing diagnostics of a completed segment-batched native language model (THEORY §413).  Forward only.

For the saved final weights of a language_batched result, on E64 windows of the first --chars DEV characters:
  * routing sharpness per layer: mean max pi, mean entropy (bits), fraction of races with max pi > .9, and unit usage
    (fraction of wins per unit);
  * bpc with the model's sampled races (evaluation seed 314159, as reported), with argmax routing (every clock noise
    set to 1, so the earliest clock is the highest score; delays then use T = 1/max rate, a labelled diagnostic), and
    with the probability mixture over --samples independent race seeds (the Jensen gap of sampled routing).
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import language_batched_benchmark as L  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402


@contextmanager
def argmax_routing():
    original = torch.Tensor.exponential_
    torch.Tensor.exponential_ = lambda self, *a, **k: self.fill_(1.)
    try:
        yield
    finally:
        torch.Tensor.exponential_ = original


def windows(text, S):
    half = S // 2
    return half, list(range(0, len(text) - S - 1, half))


@torch.no_grad()
def probabilities(model, text, S, seed, lanes, record=None):
    half, starts = windows(text, S); out = []
    for b in range(0, len(starts), lanes):
        chunk = starts[b:b + lanes]
        z = batched_logits(model, L.rows_of(text, chunk, S), seed, record=record, all_logits=True)
        out.append(torch.softmax(z.double(), -1))
    return torch.cat(out), starts, half


def bpc(prob, text, starts, half, S):
    bits = 0.; n = 0
    for i, s in enumerate(starts):
        y = torch.as_tensor(text[s + 1:s + S + 1].astype(np.int64))
        p = prob[i].gather(-1, y[:, None]).squeeze(-1)
        part = p if s == 0 else p[half:]
        bits -= float(torch.log2(part).sum()); n += part.numel()
    return bits / n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True); p.add_argument('--out', required=True)
    p.add_argument('--chars', type=int, default=50_000); p.add_argument('--samples', type=int, default=4)
    p.add_argument('--lanes', type=int, default=16, help='evaluation lanes (memory only; results do not depend on it)')
    a = p.parse_args()
    out = ROOT / a.out
    if out.exists():
        raise ValueError('Preserve existing result')
    r = json.loads((ROOT / a.result).read_text()); args = r['args']
    torch.set_num_threads(1)
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=27, classes=27, payload=args['payload'],
                                            depth=args['depth'], heads=args['heads'], pool=args['pool'])
    model.load_state_dict(torch.load(ROOT / r['final_weights'])); model.eval()
    text = L.load_text(90_000_000, a.chars); S, lanes = args['segment'], a.lanes
    record = []
    sampled, starts, half = probabilities(model, text, S, 314159, lanes, record)
    D, H, U = model.depth, model.heads, model.pool
    pis = [torch.softmax(x.double(), -1) for x in record]          # race order: step, depth, head (per lane chunk)
    layers = []
    for d in range(D):
        sel = torch.cat([pis[i] for i in range(len(pis)) if (i // H) % D == d])
        mx = sel.max(-1).values; ent = -(sel * sel.clamp_min(1e-300).log2()).sum(-1)
        layers.append(dict(depth=d, mean_max_pi=float(mx.mean()), mean_entropy_bits=float(ent.mean()),
                           max_entropy_bits=math.log2(U), frac_max_pi_above_0_9=float((mx > .9).float().mean()),
                           mean_pi_per_unit=sel.mean(0).tolist()))
    with argmax_routing():
        greedy, _, _ = probabilities(model, text, S, 314159, lanes)
    mixture = sampled.clone()
    for k in range(1, a.samples):
        mixture += probabilities(model, text, S, 314159 + 7919 * k, lanes)[0]
    mixture /= a.samples
    res = dict(result=a.result, chars=a.chars, segment=S, routing=layers,
               bpc=dict(sampled=bpc(sampled, text, starts, half, S), argmax_routing=bpc(greedy, text, starts, half, S),
                        **{f'mixture_{a.samples}_seeds': bpc(mixture, text, starts, half, S)}),
               scope='forward-only diagnostic on DEV text8[90M:90M+chars]; argmax routing also changes delays (T = 1/max rate)')
    out.write_text(json.dumps(res, indent=2) + '\n'); print(json.dumps(res['bpc']), json.dumps([(l['depth'], round(l['mean_max_pi'], 3)) for l in layers]))


if __name__ == '__main__':
    main()
