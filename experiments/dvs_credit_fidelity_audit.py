"""Frozen race-credit fidelity by depth on trained DVS checkpoints (THEORY §399).  Read-only, no optimizer step.

For sampled races in development gestures, compare the counterfactual surrogate's gradient on that race's clock
scores (TemporalRoute.backward, captured by a hook) with the exact expected-loss gradient for that single race:
force each candidate to win while every other race keeps its noise, giving loss L_i, then
d E[L] / d s_i = pi_i (L_i - sum_j pi_j L_j).  Reported per depth: sign agreement of the winner-versus-rest
contrast, cosine, and median magnitude ratio |surrogate| / |exact|.  Pool-2 races give collinear 2-vectors, so
sign agreement is the informative statistic there.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel  # noqa: E402


def run_episode(model, row, seed, force=None, capture=None):
    """force = (race index, candidate) to override one winner; capture = list collecting (depth, scores) per race."""
    counter = [0]; depth_of = []
    original = ParallelHeadRaceLanguageModel.race

    def race(scores, values=None):
        r = counter[0]; counter[0] += 1
        if force is not None and r == force[0]:
            rates = scores.to(torch.float64).exp()
            times = torch.empty_like(rates).exponential_() / rates           # same RNG consumption as the real race
            i = force[1]; t = times[i]
            return values[i], .001 + .010 * t / (1 + t), torch.tensor(i)
        if capture is not None:
            capture.append(scores)
        return original(scores, values)
    model.race = race
    try:
        model.train(); state = model.new_state()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for timestamp, content in row['events']:
                logits, _ = model.consume_event(0, timestamp, content, state)
    finally:
        del model.race
    return F.cross_entropy(logits[None], torch.tensor([row['target']])), counter[0]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--results', nargs='+', required=True)
    p.add_argument('--episodes', type=int, default=6); p.add_argument('--races-per-episode', type=int, default=24)
    a = p.parse_args()
    out = ROOT / 'experiments/results/diagnostics' / f'{a.tag}.json'
    if out.exists():
        raise ValueError('preserve prior result')
    torch.set_num_threads(1); started = time.perf_counter(); report = {}
    for path in a.results:
        res = json.loads((ROOT / path).read_text()); args = SimpleNamespace(**res['args'])
        ck = torch.load(ROOT / path.replace('.json', '.progress.pt'), weights_only=False)
        model = C.make_model(args, fast=False); model.load_state_dict(ck['best_state'])
        _, dev, _ = N.load(args)
        per_layer = model.heads; rng = np.random.default_rng(0); rows = []
        for row in dev[:a.episodes]:
            seed = 271828 + row['index']; scores_list = []
            loss, races = run_episode(model, row, seed, capture=scores_list)
            for s in scores_list:
                s.retain_grad()
            model.zero_grad(); loss.backward()
            chosen = rng.choice(races, size=min(a.races_per_episode, races), replace=False)
            for r in chosen:
                s = scores_list[r]; g = s.grad.detach().double(); pi = torch.softmax(s.detach().double(), 0)
                with torch.no_grad():
                    L = torch.tensor([float(run_episode(model, row, seed, force=(int(r), i))[0]) for i in range(len(pi))],
                                     dtype=torch.float64)
                exact = pi * (L - (pi * L).sum())
                depth = (int(r) // per_layer) % model.depth
                cos = float(F.cosine_similarity(g[None], exact[None]).item()) if exact.norm() > 0 and g.norm() > 0 else 0.
                w = int(pi.argmax())
                rows.append(dict(depth=depth, cosine=cos, sign_agree=bool(np.sign(float(g[w] - g.mean())) == np.sign(float(exact[w] - exact.mean()))),
                                 ratio=float(g.norm() / exact.norm()) if exact.norm() > 0 else None, exact_norm=float(exact.norm())))
        by = {}
        for d in range(model.depth):
            sub = [x for x in rows if x['depth'] == d and x['exact_norm'] > 1e-9]
            ratios = [x['ratio'] for x in sub if x['ratio'] is not None]
            by[str(d)] = dict(races=len(sub), sign_agreement=float(np.mean([x['sign_agree'] for x in sub])) if sub else None,
                              mean_cosine=float(np.mean([x['cosine'] for x in sub])) if sub else None,
                              median_magnitude_ratio=float(np.median(ratios)) if ratios else None)
        report[Path(path).stem] = dict(args=res['args'], final_accuracy=res['final']['accuracy'], by_depth=by)
        print(json.dumps({Path(path).stem: by}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(status='completed', models=report, episodes=a.episodes, wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen read-only single-race counterfactual audit on development gestures; exact for one race with other '
              'races\' noise fixed; not a whole-route gradient.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
