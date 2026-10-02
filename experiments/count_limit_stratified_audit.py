"""Where do counts run out?  Stratified frozen audit of count-carrying models (THEORY §393).

Counting models are near-optimal where contexts have ample evidence; a learner must show its advantage where
they run out.  This audit scores saved checkpoints position by position on the shared development window and
stratifies by the evidence the counts actually had: n_K, the prequential count (fit + development prefix) of
the top-order context at that position (0 = unseen, 1-2, 3-9, >= 10).  Count references (stream-adaptive
Witten-Bell and frozen Kneser-Ney at their dev-selected orders) are scored on the same positions.

Read-only: selected weights from each run's checkpoint, frozen, no optimizer step.  Position losses use the
same race-noise seed and chunking as the training driver's development evaluation.
"""
import argparse
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import count_reference_language as R  # noqa: E402
from e120_shared_tasks import text_slice  # noqa: E402
from sleeping_machines.count_carrying_language import CountCarryingNativeModel, eval_stream_counts  # noqa: E402
from sleeping_machines.count_escape_gate import GatedCountCarryingNativeModel  # noqa: E402

STRATA = (('unseen', 0, 0), ('1-2', 1, 2), ('3-9', 3, 9), ('>=10', 10, 10 ** 9))


def model_losses(path, dev):
    res = json.loads(path.read_text()); a = res['args']
    ck = torch.load(path.with_suffix('.progress.pt'), weights_only=False)
    if a.get('escape_gate') or a.get('count_message'):
        m = GatedCountCarryingNativeModel(a['payload'], a['depth'], a['pool'], heads=a['heads'], orders=a['orders'],
                                          escape_gate=a.get('escape_gate', False), count_message=a.get('count_message', False))
    else:
        m = CountCarryingNativeModel(a['payload'], a['depth'], a['pool'], heads=a['heads'], orders=a['orders'])
    m.load_state_dict(ck['best_state'])
    train = text_slice(0, a['fit'])
    m.register_stream('dev', eval_stream_counts(np.array(train), np.array(dev), a['orders'])); m.use_stream('dev')
    tokens = torch.tensor(dev); out = []
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(314159); m.eval(); state = m.new_state()
        for s in range(0, len(tokens) - 1, a['chunk']):
            e = min(s + a['chunk'], len(tokens) - 1)
            z, state = m.forward_chunk(tokens[s:e], state)
            out.append(F.cross_entropy(z, tokens[s + 1:e + 1], reduction='none') / math.log(2))
    return torch.cat(out).numpy(), a, res['final']['dev']['bpc']


def count_losses(fit, dev, K, method, adaptive):
    tables = R.Counts(K); tables.add_sequence(fit)
    kn = R.continuation_tables(fit, K) if method == 'kn' else None
    out = []
    for i in range(1, len(dev)):
        hist = dev[max(0, i - K):i]
        out.append(-math.log2(R.predict(tables, hist, K, method, .75, kn)[dev[i]]))
        if adaptive:
            tables.add(hist, dev[i])
    return np.array(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--results', nargs='+', required=True)
    p.add_argument('--strata-order', type=int, default=4)
    a = p.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / f'{a.tag}.json'
    if out.exists():
        raise ValueError('preserve prior result')
    torch.set_num_threads(1); started = time.perf_counter()
    dev = text_slice(90_000_000, 8192)
    rows, evidence = {}, {}
    for r in a.results:
        losses, args, saved = model_losses(ROOT / r, dev)
        fit = text_slice(0, args['fit'])
        if args['fit'] not in evidence:
            c = eval_stream_counts(np.array(fit), np.array(dev), a.strata_order)[a.strata_order - 1].sum(-1)[:len(dev) - 1]
            evidence[args['fit']] = c
            rows[f"counts_stream_adaptive_wb_o4_{args['fit']}"] = dict(fit=args['fit'], losses=count_losses(fit, dev, 4, 'wb', True))
            rows[f"counts_frozen_kn_o5_{args['fit']}"] = dict(fit=args['fit'], losses=count_losses(fit, dev, 5, 'kn', False))
        name = Path(r).stem
        rows[name] = dict(fit=args['fit'], losses=losses, saved_dev_bpc=saved, payload=args['payload'], depth=args['depth'])
        print(json.dumps(dict(scored=name, mean=float(losses.mean()), saved=saved)), flush=True)
    summary = {}
    for name, row in rows.items():
        ev = evidence[row['fit']]; L = row['losses']
        summary[name] = dict(fit=row['fit'], mean_bpc=float(L.mean()), **{k: v for k, v in row.items() if k not in ('losses', 'fit')},
                             strata={s: dict(positions=int(((ev >= lo) & (ev <= hi)).sum()),
                                             bpc=float(L[(ev >= lo) & (ev <= hi)].mean()) if ((ev >= lo) & (ev <= hi)).any() else None)
                                     for s, lo, hi in STRATA})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(status='completed', strata_order=a.strata_order, development=[90_000_000, 90_008_192],
        summary=summary, wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen read-only stratified audit; strata by prequential top-order context evidence; one seed per model.'),
        indent=2) + '\n')


if __name__ == '__main__':
    main()
