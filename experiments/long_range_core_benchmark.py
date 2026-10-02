"""Long-range learnability of the unchanged native temporal core (labelled diagnostic; THEORY §389 attribution).

Streams over the 27-symbol alphabet whose targets depend on content far back, while every local n-gram is
uninformative by construction: filler symbols 0..23 are i.i.d. uniform, so any count table of any order K
predicts a target at chance (log2 24 = 4.585 bits) unless the exact (K-context, target) pair recurs by accident.

  lag L        ... x_{t-L} ... [24] y   with y = x_{t-L}  (L counted back from the cue; delay/timing memory)
  induction W  ... q a ...      [25] q y with y = the symbol that followed q's most recent occurrence within W
                                         (content-addressed recall)

Cues arrive after random gaps.  The model is the integrated NativeStreamLanguageModel (races, addressed
persistent state, learned delays, counterfactual route credit) trained as a stream language model on every
position with per-chunk truncated credit; only the score is split into target positions and filler.
Width/depth set the full versus minimal core.  --model tapped adds learned dilated delay taps (THEORY §391).  --model kv adds the thesis's race attention over stored
keys/values (ParallelHeadRaceLanguageModel: per-position KV bank, bounded hashed candidates + recent entries,
hard race retrieval with counterfactual credit), which the native core omits.  Exploratory: one seed, synthetic, no FLOP audit (parameters,
targets and wall time recorded), not a language benchmark.
"""
import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleeping_machines.native_stream_language import NativeStreamLanguageModel  # noqa: E402
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel  # noqa: E402
from sleeping_machines.dilated_delay_taps import TappedNativeStreamLanguageModel  # noqa: E402
from sleeping_machines.context_addressed_memory import ContextAddressedNativeModel  # noqa: E402

FILLER, LAG_CUE, INDUCTION_CUE = 24, 24, 25


def make_stream(task, n, distance, seed, gap=(6, 18)):
    """tokens (n,), target mask (n,): mask[p] marks that tokens[p] is a long-range target (predicted at p-1)."""
    rng = np.random.default_rng(seed)
    toks, mask = [], []
    while len(toks) < n:
        for _ in range(int(rng.integers(gap[0], gap[1] + 1))):
            toks.append(int(rng.integers(FILLER))); mask.append(False)
        if task == 'lag':
            if len(toks) < distance:
                continue
            y = toks[-distance]
            toks += [LAG_CUE, y]; mask += [False, True]
        else:
            window = toks[-distance:]
            pairs = {window[i]: window[i + 1] for i in range(len(window) - 1)
                     if window[i] < FILLER and window[i + 1] < FILLER}
            if not pairs:
                continue
            q = int(rng.choice(sorted(pairs)))
            toks += [INDUCTION_CUE, q, pairs[q]]; mask += [False, False, True]
    return np.array(toks[:n], np.int64), np.array(mask[:n], bool)


def score(model, tokens, mask, chunk):
    with torch.no_grad(), torch.random.fork_rng():
        torch.manual_seed(314159); model.eval(); state = model.new_state(); losses, hits = [], []
        for s in range(0, len(tokens) - 1, chunk):
            e = min(s + chunk, len(tokens) - 1)
            z, state = model.forward_chunk(tokens[s:e], state)
            losses.append(F.cross_entropy(z, tokens[s + 1:e + 1], reduction='none'))
            hits.append(z.argmax(-1) == tokens[s + 1:e + 1])
        loss, hit, m = torch.cat(losses), torch.cat(hits), torch.as_tensor(mask[1:])
    bits = 1 / math.log(2)
    return dict(all_bpc=float(loss.mean()) * bits, target_bpc=float(loss[m].mean()) * bits,
                filler_bpc=float(loss[~m].mean()) * bits, target_accuracy=float(hit[m].float().mean()),
                targets=int(m.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--task', choices=('lag', 'induction', 'text'), required=True)
    p.add_argument('--distance', type=int, default=0); p.add_argument('--fit', type=int, default=8192)
    p.add_argument('--dev', type=int, default=4096); p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--chunk', type=int, default=16); p.add_argument('--payload', type=int, default=16)
    p.add_argument('--depth', type=int, default=8); p.add_argument('--heads', type=int, default=2)
    p.add_argument('--pool', type=int, default=2); p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--model', choices=('native', 'kv', 'tapped', 'addressed'), default='native')
    p.add_argument('--order', type=int, default=3); p.add_argument('--buckets', type=int, default=4096)
    p.add_argument('--matching', type=int, default=8); p.add_argument('--recent', type=int, default=4)
    a = p.parse_args()
    out = ROOT / 'experiments/results/long_range_core' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required; prior results are preserved')
    torch.set_num_threads(1); torch.manual_seed(a.seed); started = time.perf_counter()
    if a.task == 'text':  # shared language protocol: text8[0:fit] and the 8,191-target development window
        sys.path.insert(0, str(ROOT / 'experiments'))
        from e120_shared_tasks import text_slice
        fit, dev = np.array(text_slice(0, a.fit), np.int64), np.array(text_slice(90_000_000, a.dev), np.int64)
        fit_mask, dev_mask = np.ones(len(fit), bool), np.ones(len(dev), bool)
    else:
        fit, fit_mask = make_stream(a.task, a.fit, a.distance, a.seed)
        dev, dev_mask = make_stream(a.task, a.dev, a.distance, a.seed + 10_000)
    fit_t, dev_t = torch.tensor(fit), torch.tensor(dev)
    model = (NativeStreamLanguageModel(a.payload, a.depth, a.pool, a.heads) if a.model == 'native' else
             TappedNativeStreamLanguageModel(a.payload, a.depth, a.pool, a.heads) if a.model == 'tapped' else
             ContextAddressedNativeModel(a.payload, a.depth, a.pool, a.heads, order=a.order, buckets=a.buckets)
             if a.model == 'addressed' else
             ParallelHeadRaceLanguageModel(a.payload, a.depth, a.pool, matching=a.matching, recent=a.recent, heads=a.heads))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sources = ['experiments/long_range_core_benchmark.py', 'sleeping_machines/native_stream_language.py',
               'sleeping_machines/parallel_head_race_language.py', 'sleeping_machines/dilated_delay_taps.py',
               'sleeping_machines/context_addressed_memory.py']
    result = dict(status='running', args=vars(a), parameters=sum(q.numel() for q in model.parameters()),
                  chance_target_bpc=math.log2(FILLER), initial_dev=score(model, dev_t, dev_mask, a.chunk), curve=[],
                  data_sha256=hashlib.sha256(fit.tobytes() + dev.tobytes()).hexdigest(),
                  source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
                  hardware=dict(device='cpu', threads=1, platform=platform.platform(), torch=torch.__version__),
                  protocol='stream LM loss on every position; per-chunk backward, detach and Adam step; '
                           'persistent state carried across chunks; dev scored with frozen weights from a cold state',
                  scope='Labelled synthetic diagnostic: long-range learnability of the native core; counts are at '
                        'chance on targets by construction. One seed; no FLOP audit; not language evidence.')
    print(json.dumps(dict(started=a.tag, parameters=result['parameters'], initial=result['initial_dev'])), flush=True)
    for epoch in range(1, a.epochs + 1):
        model.train(); state = model.new_state(); total = 0.
        for s in range(0, a.fit - 1, a.chunk):
            e = min(s + a.chunk, a.fit - 1)
            opt.zero_grad(set_to_none=True)
            z, state = model.forward_chunk(fit_t[s:e], state)
            loss = F.cross_entropy(z, fit_t[s + 1:e + 1])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite fitting loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            state = state.detach(); total += float(loss) * (e - s)
        row = dict(tap_delays=model.tap_delays() if a.model == 'tapped' else None, epoch=epoch, fitting_bpc=total / (a.fit - 1) / math.log(2), dev=score(model, dev_t, dev_mask, a.chunk),
                   wall_s=time.perf_counter() - started)
        result['curve'].append(row); print(json.dumps(row), flush=True)
    result.update(status='completed', final=result['curve'][-1],
                  wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
