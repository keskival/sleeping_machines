"""Long-range learnability of the unchanged native temporal core (labelled diagnostic; THEORY §389 attribution).

Streams over the 27-symbol alphabet whose targets depend on content far back.
Filler symbols 0..23 are i.i.d. uniform, but inserted cues and copied targets are
not. Local count controls must be measured: there is no all-orders chance proof.
Lag targets can copy earlier cues; log2(24) is only a reference, not their chance loss.

  lag L        ... x_{t-L} ... [24] y   with y = x_{t-L}  (L counted back from the cue; delay/timing memory)
  induction W  ... q a ...      [25] q y with y = the symbol that followed q's most recent occurrence within W
                                         (content-addressed recall)

Cues arrive after random gaps.  The model is the integrated NativeStreamLanguageModel (races, addressed
persistent state, learned delays, counterfactual route credit) trained as a stream language model on every
position with per-chunk truncated credit; only the score is split into target positions and filler.
Width/depth set the full versus minimal core.  --model tapped adds learned dilated delay taps (THEORY §391).  --model kv adds the thesis's race attention over stored
keys/values (ParallelHeadRaceLanguageModel: per-position KV bank, bounded hashed candidates + recent entries,
hard race retrieval with counterfactual credit), which the native core omits. Exploratory one-seed
diagnostic; bounded smokes can audit every fitting operation, but long-fit accounting/recovery
admission is still open. This is not a language benchmark.
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
sys.path.insert(0, str(ROOT / 'experiments'))
from parallel_head_gradient_accumulation import GradientAccumulator  # noqa: E402
from race_language_screen import capture  # noqa: E402
from native_language_helpers import source_hashes  # noqa: E402
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
    return dict(all_bpc=float(loss.mean()) * bits, target_bpc=float(loss[m].mean()) * bits if m.any() else None,
                filler_bpc=float(loss[~m].mean()) * bits if (~m).any() else None,
                target_accuracy=float(hit[m].float().mean()) if m.any() else None,
                targets=int(m.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--task', choices=('lag', 'induction', 'text'), required=True)
    p.add_argument('--distance', type=int, default=0); p.add_argument('--fit', type=int, default=8192)
    p.add_argument('--dev', type=int, default=4096); p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--chunk', type=int, default=16); p.add_argument('--payload', type=int, default=16)
    p.add_argument('--update-targets', type=int, default=64)
    p.add_argument('--audit-work', action='store_true', help='Audit every fitting operation in a bounded smoke')
    p.add_argument('--depth', type=int, default=8); p.add_argument('--heads', type=int, default=2)
    p.add_argument('--pool', type=int, default=2); p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--model', choices=('native', 'kv', 'tapped', 'addressed'), default='native')
    p.add_argument('--order', type=int, default=3); p.add_argument('--buckets', type=int, default=4096)
    p.add_argument('--matching', type=int, default=8); p.add_argument('--recent', type=int, default=4)
    a = p.parse_args()
    if a.chunk < 1 or a.update_targets < a.chunk or a.update_targets % a.chunk:
        raise ValueError('Positive credit length must divide the separate optimizer interval')
    if a.fit < 2 or a.dev < 2 or a.epochs < 1:
        raise ValueError('At least one fitting/development target and one pass required')
    if a.audit_work and (a.fit > 512 or a.dev > 512):
        raise ValueError('Whole-operation tracing is bounded to512-token accounting smokes')
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
    learner = GradientAccumulator(model, opt, a.lr)
    # Extension construction consumes random draws; it must not silently change
    # the subsequent training race-noise sequence in a paired comparison.
    torch.manual_seed(a.seed + 100_000)
    sources = ['experiments/long_range_core_benchmark.py', 'sleeping_machines/native_stream_language.py',
               'sleeping_machines/parallel_head_race_language.py', 'sleeping_machines/dilated_delay_taps.py',
               'sleeping_machines/context_addressed_memory.py', 'sleeping_machines/addressed_event_heads.py',
               'sleeping_machines/sparse_race_language.py', 'sleeping_machines/parallel_stream_language.py',
               'sleeping_machines/packed_episodic_race_language.py',
               'sleeping_machines/indexed_episodic_race_language.py',
               'experiments/parallel_head_gradient_accumulation.py']
    if a.task == 'text': sources.append('experiments/e120_shared_tasks.py')
    result = dict(status='running', args=vars(a), parameters=sum(q.numel() for q in model.parameters()),
                  uniform_24_reference_bpc=math.log2(FILLER),
                  uniform_27_reference_bpc=math.log2(27),
                  observed_dev_target_symbols=np.unique(dev[dev_mask]).tolist(),
                  target_fraction=float(fit_mask[1:].mean()),
                  initial_dev=score(model, dev_t, dev_mask, a.chunk), curve=[],
                  data_sha256=hashlib.sha256(fit.tobytes() + dev.tobytes()).hexdigest(),
                  source_sha256={**source_hashes(), **{s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources}},
                  hardware=dict(device='cpu', threads=1, platform=platform.platform(), torch=torch.__version__),
                  protocol='stream LM loss on every position; per-credit-chunk backward/detach; '
                           'target-weighted gradients normalized/clipped once per separate optimizer window; '
                           'persistent state carried across chunks; dev scored with frozen weights from a cold state',
                  scope=('Labelled synthetic memory diagnostic; local controls require measurement. '
                         if a.task != 'text' else 'Exploratory text development diagnostic. ') +
                        ('One seed; whole-operation smoke accounting only, no long-fit recovery admission. '
                         if a.audit_work else 'One seed; no FLOP audit or recovery admission. ') +
                        'Do not admit long fits from this driver yet.')
    print(json.dumps(dict(started=a.tag, parameters=result['parameters'], initial=result['initial_dev'])), flush=True)
    work_records=[]; storage=[]
    for epoch in range(1, a.epochs + 1):
        model.train(); state = model.new_state(); total = 0.
        for window in range(0, a.fit - 1, a.update_targets):
            window_end = min(window + a.update_targets, a.fit - 1)
            for s in range(window, window_end, a.chunk):
                e = min(s + a.chunk, window_end)
                if a.audit_work:
                    box={}
                    def micro():
                        box['result']=learner.accumulate(fit_t[s:e], fit_t[s + 1:e + 1], state)
                    work_records.append(capture(micro))
                    loss,state,_=box['result']
                else:
                    loss, state, _ = learner.accumulate(fit_t[s:e], fit_t[s + 1:e + 1], state)
                total += loss
            if a.audit_work:work_records.append(capture(learner.update))
            else:learner.update()
        storage.append(state.packed_storage() if hasattr(state,'packed_storage') else state.storage())
        row = dict(tap_delays=model.tap_delays() if a.model == 'tapped' else None, epoch=epoch, fitting_bpc=total / (a.fit - 1) / math.log(2), dev=score(model, dev_t, dev_mask, a.chunk),
                   optimizer_updates=learner.updates, fitted_targets=learner.total_targets,
                   wall_s=time.perf_counter() - started)
        result['curve'].append(row); print(json.dumps(row), flush=True)
    result['state_storage_by_pass']=storage
    if a.audit_work:
        model.eval();state=model.new_state();warm=min(128,a.dev-2)
        with torch.no_grad(),torch.random.fork_rng():
            torch.manual_seed(314159);_,state=model.forward_chunk(dev_t[:warm],state)
            inference_targets=min(a.chunk,a.dev-1-warm)
            def inference():
                z,_=model.forward_chunk(dev_t[warm:warm+inference_targets],state)
                F.cross_entropy(z,dev_t[warm+1:warm+inference_targets+1],reduction='sum')
            infer=capture(inference)
        arithmetic=sum(r['arithmetic_flops'] for r in work_records)
        special=sum(r['special_function_evaluations'] for r in work_records)
        result['work']=dict(fitting_targets=learner.total_targets,optimizer_steps=learner.updates,
            cpu_whole_fit_arithmetic_flops=arithmetic,cpu_whole_fit_special_function_evaluations=special,
            cpu_whole_fit_unit_special_flops=arithmetic+special,
            cpu_fit_unit_special_flops_per_target=(arithmetic+special)/learner.total_targets,
            inference_targets=inference_targets,inference_warm_tokens=warm,
            cpu_inference_unit_special_flops_per_target=(infer['arithmetic_flops']+infer['special_function_evaluations'])/inference_targets,
            formula_coverage_complete=all(r['formula_coverage_complete'] for r in work_records+[infer]),
            traces=work_records,inference_trace=infer,
            scope='Actual whole-smoke forward/loss/backward/normalization/clipping/Adam CPU-emulator arithmetic, '
                  '2FLOPs/MAC plus unit-weight specials. Dev passes, RNG, integer/hash operations, Python metadata '
                  'and traffic separate; no physical timing or energy projection.')
    result.update(status='completed', final=result['curve'][-1],
                  wall_s=time.perf_counter() - started, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
