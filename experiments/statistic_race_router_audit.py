"""Frozen router audit for statistic-valued race memory (THEORY §392): why do writes collapse?

Loads the selected weights of a completed pooled run, streams the development window with frozen weights from
the saved fit-pass pooled counts, and records the race distribution pi at each position: entropy, max share,
argmax histogram, effective number of receivers exp(H), and the pooled level's share of the cascade's base
mass. No optimizer step; read-only.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from e120_shared_tasks import text_slice  # noqa: E402
from sleeping_machines.count_carrying_language import eval_stream_counts, fit_stream_counts  # noqa: E402
from sleeping_machines.statistic_race_memory import StatisticRaceNativeModel  # noqa: E402


def main():
    p = argparse.ArgumentParser(); p.add_argument('--result', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--positions', type=int, default=1024)
    a = p.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / f'{a.tag}.json'
    if out.exists():
        raise ValueError('preserve prior result')
    torch.set_num_threads(1); started = time.perf_counter()
    res = json.loads((ROOT / a.result).read_text()); args = res['args']
    ck = torch.load(ROOT / a.result.replace('.json', '.progress.pt'), weights_only=False)
    model = StatisticRaceNativeModel(args['payload'], args['depth'], args['pool'], heads=args['heads'],
                                     orders=args['orders'], addresses=args['pool_addresses'],
                                     key_dim=args['pool_key_dim'], escape_gate=args['escape_gate'],
                                     count_message=args['count_message'])
    model.load_state_dict(ck['best_state'])
    train = text_slice(0, args['fit']); dev = text_slice(90_000_000, args['dev'])
    model.register_stream('dev', eval_stream_counts(np.array(train), np.array(dev), args['orders'])); model.use_stream('dev')
    model.eval(); model.seed_pool = True
    pis = []
    original = model.race_distribution
    def record(h):
        pi = original(h); pis.append(pi.detach()); return pi
    model.race_distribution = record
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(314159); state = model.new_state()
        tokens = torch.tensor(dev[:a.positions + 1])
        for s in range(0, a.positions, 16):
            _, state = model.forward_chunk(tokens[s:s + 16], state)
    pi = torch.stack(pis).double()
    ent = -(pi * pi.clamp_min(1e-300).log()).sum(-1)
    arg = pi.argmax(-1)
    hist = np.bincount(arg.numpy(), minlength=pi.shape[1])
    seed_occ = int((model.pool_seed.sum(-1) > 0).sum())
    scores = (model.pool_keys @ model.pool_query.weight).norm(dim=-1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        status='completed', result=a.result, positions=int(pi.shape[0]), addresses=int(pi.shape[1]),
        entropy_nats_mean=float(ent.mean()), uniform_entropy_nats=math.log(pi.shape[1]),
        effective_receivers_mean=float(ent.exp().mean()), max_share_mean=float(pi.max(-1).values.mean()),
        argmax_distinct=int((hist > 0).sum()), argmax_top5_share=float(np.sort(hist)[::-1][:5].sum() / hist.sum()),
        seed_occupied_receivers=seed_occ, temperature=float(torch.exp(model.pool_log_temperature)),
        key_query_gain_norm_mean=float(scores.mean()), key_query_gain_norm_max=float(scores.max()),
        wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen read-only router audit; development positions from saved fit-pass counts; no optimizer.'),
        indent=2) + '\n')


if __name__ == '__main__':
    main()
