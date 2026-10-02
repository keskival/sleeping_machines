"""Frozen audit of commit-aware race credit (THEORY §401) against exact single-race gradients.  Read-only.

The reference AddressedEventHeads event step is instrumented (forward values unchanged): per race it records every
unit's proposed value and proposed memory, the old slot contents, the delivered value and the scores; after every
event each memory slot is replaced by an identity alias, so a slot's gradient counts only reads after that event
(G_u).  Estimators compared with the exact forced-winner gradient pi_i (L_i - sum_j pi_j L_j):

  surrogate      the trained model's counterfactual teacher (captured score gradient);
  exact_pi       pi_i (g.v_i - sum_j pi_j g.v_j)                   (value path only, §400);
  commit         pi_i (l_i - sum_j pi_j l_j), l_i = g.v_i + G_i.(m_i^new - m_i^old)   (§401).
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
import dvs_credit_fidelity_audit as A  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402


def instrumented_step(model, source, timestamp, content, state, log):
    """copy of AddressedEventHeads.consume_event (training path) with recording and post-event slot aliasing."""
    mark = torch.as_tensor(content, dtype=model.embedding.weight.dtype)
    x = model.embedding.weight[source] + model.content(mark)
    arrival = torch.tensor(float(timestamp), dtype=torch.float64)
    if source in state.contexts:
        previous, times = state.contexts[source]
        read_time = torch.maximum(arrival, times.max()); arrival = read_time
        context = model.align(list(previous.split(model.payload)), times, arrival, model.depth - 1)
        x = F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context, (model.total_payload,))
    for depth in range(model.depth):
        mixed = model.channel_mix[depth](x)
        all_features = F.layer_norm(mixed, (model.total_payload,))
        values, arrivals = [], []
        for head in range(model.heads):
            incoming = mixed[head * model.payload:(head + 1) * model.payload]
            units = model.units[depth][head][source]
            query = model.queries[depth][head](all_features)
            keys = [(depth, head, source, i) for i in range(len(units))]
            olds = [state.memories.get(k, incoming.new_zeros(model.payload)) for k in keys]
            scores = torch.stack([(query @ (u.key + u.key_read(m))) / math.sqrt(model.payload) + u.clock_bias
                                  for u, m in zip(units, olds)]).clamp(-12, 12)
            proposals = [u.propose(incoming, m, state.arrivals.get(k), arrival) for u, m, k in zip(units, olds, keys)]
            value, delay, winner = model.race(scores, torch.stack([p[0] for p in proposals]))
            index = int(winner)
            if value.requires_grad:
                value.retain_grad()
            log.append(dict(scores=scores, values=torch.stack([p[0] for p in proposals]).detach(),
                            new=torch.stack([p[1] for p in proposals]).detach(), old=torch.stack(olds).detach(),
                            delivered=value, keys=keys, winner=index))
            state.memories[keys[index]] = proposals[index][1]; state.arrivals[keys[index]] = arrival
            state.selected_updates += 1; values.append(value); arrivals.append(arrival + delay)
        arrival = torch.stack(arrivals).max()
        x = model.align(values, arrivals, arrival, depth)
    state.contexts[source] = (torch.cat(values), torch.stack(arrivals))
    state.events += 1; state.last_input_time = float(timestamp)
    aliases = {}
    for k, m in list(state.memories.items()):          # future-only slot gradients: alias every slot after the event
        a = m.clone(); a.retain_grad(); state.memories[k] = a; aliases[k] = a
    log.append(dict(aliases=aliases))
    return model.head(x)


def instrumented_episode(model, row, seed):
    log = []; model.train(); state = model.new_state()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for timestamp, content in row['events']:
            logits = instrumented_step(model, 0, timestamp, content, state, log)
    return F.cross_entropy(logits[None], torch.tensor([row['target']])), log


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
        _, dev, _ = N.load(args); rng = np.random.default_rng(0); rows = []
        for row in dev[:a.episodes]:
            seed = 271828 + row['index']
            loss, log = instrumented_episode(model, row, seed)
            races = [r for r in log if 'scores' in r]
            for r in races:
                r['scores'].retain_grad()
            model.zero_grad(); loss.backward()
            # future-only slot gradient of each race's slots: the aliases created after that race's event
            alias_after, current = [], None
            for entry in reversed(log):
                if 'aliases' in entry:
                    current = entry['aliases']
                else:
                    alias_after.append(current)
            alias_after.reverse()
            chosen = rng.choice(len(races), size=min(a.races_per_episode, len(races)), replace=False)
            for ri in chosen:
                r = races[int(ri)]; aliases = alias_after[int(ri)]
                pi = torch.softmax(r['scores'].detach().double(), 0)
                g = r['delivered'].grad.detach().double() if r['delivered'].grad is not None else torch.zeros(r['values'].shape[1], dtype=torch.float64)
                gv = r['values'].double() @ g
                G = torch.stack([(aliases[k].grad if aliases is not None and k in aliases and aliases[k].grad is not None
                                  else torch.zeros(r['new'].shape[1])).double() for k in r['keys']])
                commit = (G * (r['new'].double() - r['old'].double())).sum(-1)
                ell_v, ell_c = gv, gv + commit
                est_v = pi * (ell_v - (pi * ell_v).sum()); est_c = pi * (ell_c - (pi * ell_c).sum())
                sur = r['scores'].grad.detach().double()
                with torch.no_grad():
                    L = torch.tensor([float(A.run_episode(model, row, seed, force=(int(ri), i))[0]) for i in range(len(pi))],
                                     dtype=torch.float64)
                exact = pi * (L - (pi * L).sum())
                if exact.norm() < 1e-9:
                    continue
                w = int(pi.argmax()); depth = (int(ri) // model.heads) % model.depth
                def stats(e):
                    cos = float(F.cosine_similarity(e[None], exact[None]).item()) if e.norm() > 0 else 0.
                    return cos, bool(np.sign(float(e[w] - e.mean())) == np.sign(float(exact[w] - exact.mean()))), float(e.norm() / exact.norm())
                rows.append(dict(depth=depth, surrogate=stats(sur), exact_pi=stats(est_v), commit=stats(est_c),
                                 commit_share=float(commit.abs().sum() / (gv.abs().sum() + commit.abs().sum() + 1e-30))))
        by = {}
        for d in range(model.depth):
            sub = [x for x in rows if x['depth'] == d]
            by[str(d)] = dict(races=len(sub), **{f'{name}_{stat}': (float(np.mean([x[name][j] for x in sub])) if stat != 'median_magnitude_ratio'
                                                               else float(np.median([x[name][j] for x in sub]))) if sub else None
                                                 for name in ('surrogate', 'exact_pi', 'commit')
                                                 for j, stat in ((0, 'mean_cosine'), (1, 'sign_agreement'), (2, 'median_magnitude_ratio'))},
                              mean_commit_share=float(np.mean([x['commit_share'] for x in sub])) if sub else None)
        report[Path(path).stem] = dict(final_accuracy=res['final']['accuracy'], by_depth=by)
        print(json.dumps({Path(path).stem: by}), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(status='completed', models=report, episodes=a.episodes, wall_s=time.perf_counter() - started,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen read-only; exact single-race gradients by forced-winner replay; commit terms from post-event aliased '
              'slot gradients (future reads only); linearized estimators.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
