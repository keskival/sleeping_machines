"""Segment-batched native language training at scale (THEORY §409).  text8, 27 symbols.

The integrated native core (AddressedEventHeads, one source address, content = one-hot character, factorized race law)
is trained on independent segments of S characters with the state reset per segment, and every segment of a
window shares the same per-step race-noise seed.  Segments therefore run as lanes of one exact batched pass
(sleeping_machines/batched_episodes.py); credit spans the whole segment.  Evaluation follows the E64 window protocol:
windows of T = S characters at stride S/2, the first window scored whole and later windows on their second half.
That is the protocol of the saved LSTM/Transformer controls, scored on the same test text8[95M:95M+test].
Work is traced on sampled windows and extrapolated per character (labelled estimate).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from e120_shared_tasks import text_slice  # noqa: E402
from parallel_head_accumulated_language import merge  # noqa: E402
from race_language_screen import capture  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402

EYE = np.eye(27, dtype=np.float32)


def rows_of(text, starts, S):
    return [dict(events=[(float(t), EYE[c]) for t, c in enumerate(text[s:s + S])]) for s in starts]


LOGITS = batched_logits


def window_scores(model, text, S, seed, lanes):
    """E64 protocol: windows of S at stride S/2; first window scored whole, later windows on their second half."""
    half = S // 2; starts = list(range(0, len(text) - S - 1, half)); bits = 0.; n = 0
    model.eval()
    with torch.no_grad():
        for b in range(0, len(starts), lanes):
            chunk = starts[b:b + lanes]
            z = LOGITS(model, rows_of(text, chunk, S), seed, all_logits=True)
            y = torch.tensor(np.stack([text[s + 1:s + S + 1] for s in chunk]))
            ce = F.cross_entropy(z.reshape(-1, 27), y.reshape(-1), reduction='none').view(len(chunk), S)
            for i, s in enumerate(chunk):
                part = ce[i] if s == 0 else ce[i, half:]
                bits += float(part.sum()) / math.log(2); n += part.numel()
    return bits / n, n


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--fit', type=int, default=10_000_000)
    p.add_argument('--test', type=int, default=1_000_000); p.add_argument('--dev', type=int, default=200_000)
    p.add_argument('--segment', type=int, default=128); p.add_argument('--lanes', type=int, default=128)
    p.add_argument('--passes', type=float, default=1.); p.add_argument('--lr', type=float, default=.002)
    p.add_argument('--clip', type=float, default=1.); p.add_argument('--payload', type=int, default=16)
    p.add_argument('--depth', type=int, default=8); p.add_argument('--heads', type=int, default=2)
    p.add_argument('--pool', type=int, default=2); p.add_argument('--seed', type=int, default=6)
    p.add_argument('--eval-every', type=int, default=0, help='windows between dev evaluations (0: end only)')
    p.add_argument('--max-windows', type=int, default=0, help='stop after this many windows (throughput smoke)')
    p.add_argument('--trace-windows', type=int, default=2)
    p.add_argument('--skip-init-from', type=int, default=0, help='near-identity init for layers >= this index (§410)')
    p.add_argument('--skip-gate-bias', type=float, default=-4.)
    p.add_argument('--cosine', action='store_true', help='cosine-annealed learning rate over all windows (E64 controls)')
    p.add_argument('--eval-segment', type=int, default=0, help='also score dev/test with E64 windows of this length')
    p.add_argument('--compiled', action='store_true', help='compiled layer steps (sleeping_machines/compiled_episodes.py, §412)')
    a = p.parse_args()
    out = ROOT / 'experiments/results/language_batched' / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); torch.manual_seed(a.seed); started = time.perf_counter()
    global LOGITS
    if a.compiled:
        from torch._dynamo import config as dynamo_config
        from torch._inductor import config as inductor_config
        from sleeping_machines.compiled_episodes import compiled_logits
        inductor_config.compile_threads = 1                 # no compile-worker pool (memory floor)
        dynamo_config.cache_size_limit = 64                 # train/eval lane counts and grad modes each specialize
        LOGITS = compiled_logits
    fit = np.array(text_slice(0, a.fit), np.int64)
    dev = np.array(text_slice(90_000_000, a.dev), np.int64)
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=27, classes=27, payload=a.payload, depth=a.depth,
                                            heads=a.heads, pool=a.pool)
    if a.skip_init_from:
        from dvs_batched_large_benchmark import skip_init
        model = skip_init(model, a.skip_init_from, a.skip_gate_bias)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    S, B = a.segment, a.lanes
    rng = np.random.default_rng(a.seed + 10)
    total_windows = int(a.passes * (len(fit) - 1) // (S * B))
    if a.max_windows:
        total_windows = min(total_windows, a.max_windows)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total_windows) if a.cosine else None
    ledger = {}; traced_chars = 0; curve = []; seen = 0
    result = dict(status='running', args=vars(a), parameters=sum(q.numel() for q in model.parameters()),
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/language_batched_benchmark.py', 'sleeping_machines/batched_episodes.py',
                                  'experiments/dvs_batched_large_benchmark.py',
                                  'sleeping_machines/fast_native_core.py', 'sleeping_machines/addressed_event_heads.py',
                                  'sleeping_machines/compiled_episodes.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, device='cpu', threads=1),
                  protocol=dict(fit=[0, a.fit], dev=[90_000_000, 90_000_000 + a.dev], test=[95_000_000, 95_000_000 + a.test],
                                segment=S, lanes=B, state='reset per segment', credit='whole segment (exact BPTT within it)',
                                race='factorized law; shared per-step noise across lanes; per-window seed',
                                evaluation='E64 windows of S, stride S/2, second half scored after the first window',
                                selection='final weights (no development selection)',
                                schedule='cosine annealing over all windows' if a.cosine else 'constant learning rate',
                                kernels='compiled layer steps (torch.compile/inductor, contract-tested against the batched path)'
                                if a.compiled else 'eager batched path'))
    window_times = []
    for w in range(total_windows):
        starts = rng.integers(0, len(fit) - S - 1, B)
        rows = rows_of(fit, starts, S); y = torch.tensor(np.stack([fit[s + 1:s + S + 1] for s in starts]))
        seed = 100000 + a.seed * 1000 + w
        box = {}
        def step(logits=None):
            model.train(); opt.zero_grad(set_to_none=True)
            z = (logits or LOGITS)(model, rows, seed, all_logits=True)
            loss = F.cross_entropy(z.reshape(-1, 27), y.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip, error_if_nonfinite=True); opt.step()
            box['loss'] = float(loss.detach())
        t = time.perf_counter()
        if w < a.trace_windows:
            # traced windows run the eager batched path: fused kernels bypass the operator tracer
            rec = capture(lambda: step(batched_logits)); ledger = merge([ledger, rec]) if ledger else rec; traced_chars += S * B
        else:
            step()
        if schedule is not None:
            schedule.step()
        window_times.append(time.perf_counter() - t); seen += S * B
        if w % 50 == 0 or w == total_windows - 1:
            print(json.dumps(dict(window=w, chars=seen, train_bits=box['loss'] / math.log(2),
                                  chars_per_s=S * B / np.mean(window_times[-20:]))), flush=True)
        if a.eval_every and (w + 1) % a.eval_every == 0:
            bpc, n = window_scores(model, dev[:50_000], S, 314159, B)
            curve.append(dict(window=w + 1, chars=seen, dev50k_bpc=bpc)); print(json.dumps(curve[-1]), flush=True)
    work_per_char = (ledger['arithmetic_flops'] + ledger['special_function_evaluations']) / traced_chars if traced_chars else None
    result.update(curve=curve, fitting_chars=seen, windows=total_windows,
                  train_chars_per_s=float(S * B / np.mean(window_times[a.trace_windows:] or window_times)),
                  work=dict(fit_unit_special_flops_per_char_estimate=work_per_char,
                            whole_fit_unit_special_flops_estimate=work_per_char * seen if work_per_char else None,
                            scope='first windows fully traced, extrapolated per character; evaluation separate'))
    if not a.max_windows:
        result['dev_bpc'], result['dev_targets'] = window_scores(model, dev, S, 314159, B)
        test = np.array(text_slice(95_000_000, a.test), np.int64)
        result['test_bpc'], result['test_targets'] = window_scores(model, test, S, 314159, B)
        if a.eval_segment and a.eval_segment != S:      # same weights, E64 windows of another length (e.g. T=256)
            E = a.eval_segment; lanes = max(1, B * S // E)
            result['eval_segment'] = E
            result['dev_bpc_eval_segment'], _ = window_scores(model, dev, E, 314159, lanes)
            result['test_bpc_eval_segment'], result['test_targets_eval_segment'] = window_scores(model, test, E, 314159, lanes)
    result.update(status='completed', wall_s=time.perf_counter() - started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('dev_bpc', 'test_bpc', 'train_chars_per_s', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
