"""Evaluate a trained E64 character Transformer converted to race attention (THEORY §397).  Inference only.

Loads a saved {"args", "state"} E64 Transformer checkpoint and scores text8 from the E64 test segment start with the
E64 window protocol (windows of ctx, half-window stride, the first window scored whole, later windows their
second half).  The original network and the converted race stream score identical positions.  Delivery modes:
expected (must reproduce the original), sampled races S in {1,4,16}, shortlists m in {8,32,64}.  Reports bpc,
max |log-prob difference| to the original, key scores and value reads per token.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import e64_lm_baselines as E  # noqa: E402
from e120_shared_tasks import text_slice  # noqa: E402
from sleeping_machines.race_transformer import RaceTransformer  # noqa: E402


def windows(n, T):
    starts = list(range(0, n - T - 1, T // 2))
    return [(s, 0 if i == 0 else T // 2) for i, s in enumerate(starts)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--chars', type=int, default=20000); p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    out = ROOT / 'experiments/results/race_transformer' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise ValueError('preserve prior result')
    torch.set_num_threads(1); started = time.perf_counter()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False); args = saved['args']
    if args['model'] != 'transformer':
        raise ValueError('Transformer checkpoint required')
    net = E.TfLM(args['size'], args['layers'], args['ctx'], 0.0); net.load_state_dict(saved['state']); net.eval()
    T = args['ctx']; text = torch.tensor(text_slice(95_000_000, a.chars))
    modes = [('expected', {}), *[(f'sampled_S{s}', dict(mode='sampled', samples=s)) for s in (1, 4, 16)],
             *[(f'shortlist_m{m}', dict(mode='shortlist', shortlist=m)) for m in (8, 32, 64)]]
    rows = {name: dict(bits=0., n=0, key_scores=0, value_reads=0, max_logprob_diff=0.) for name, _ in modes}
    ref_bits, ref_n = 0., 0
    with torch.no_grad():
        for s0, skip in windows(len(text), T):
            x, y = text[s0:s0 + T], text[s0 + 1:s0 + T + 1]
            ref = F.log_softmax(net(x[None])[0][0], -1)
            ref_bits -= float(ref[skip:].gather(1, y[skip:, None]).sum()) / math.log(2); ref_n += T - skip
            for name, kw in modes:
                race = RaceTransformer(net, generator=torch.Generator().manual_seed(a.seed + s0), **kw)
                logits, state = race.forward_stream(x)
                lp = F.log_softmax(logits, -1); r = rows[name]
                r['bits'] -= float(lp[skip:].gather(1, y[skip:, None]).sum()) / math.log(2); r['n'] += T - skip
                r['key_scores'] += state.key_scores; r['value_reads'] += state.value_reads
                r['max_logprob_diff'] = max(r['max_logprob_diff'], float((lp - ref).abs().max()))
    summary = {name: dict(bpc=r['bits'] / r['n'], max_logprob_diff=r['max_logprob_diff'],
                          key_scores_per_token=r['key_scores'] / (len(windows(len(text), T)) * T),
                          value_reads_per_token=r['value_reads'] / (len(windows(len(text), T)) * T))
               for name, r in rows.items()}
    out.write_text(json.dumps(dict(status='completed', args=vars(a), checkpoint_args=args,
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(), reference_bpc=ref_bits / ref_n,
        scored_targets=ref_n, modes=summary, wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Inference-only conversion of a trained dense Transformer; work counts are logical key scores/value reads '
              'per processed token (window recomputation included), not FLOPs or energy.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
