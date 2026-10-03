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
import os
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


def load_text(start, n, chunk=10_000_000):
    """text8[start:start+n] as uint8 symbols (0-26), read in chunks: 90M characters need 90 MB, not the int64
    temporaries of one text_slice call.  Values equal text_slice's."""
    out = np.empty(n, np.uint8)
    for b in range(0, n, chunk):
        out[b:b + chunk] = text_slice(start + b, min(chunk, n - b))
    return out


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
            y = torch.tensor(np.stack([text[s + 1:s + S + 1] for s in chunk])).long()
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
    p.add_argument('--route-credit', choices=('none', 'linear', 'linear_rw'), default='none',
                   help='linear: linearized local-expectation value credit to the race scores; linear_rw: also the '
                        'linearized write-address credit (§413)')
    p.add_argument('--checkpoint-every', type=int, default=0, help='windows between exact-resume checkpoints (0: none)')
    p.add_argument('--resume', action='store_true', help='continue from this tag\'s checkpoint')
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
    fit = load_text(0, a.fit)
    dev = load_text(90_000_000, a.dev)
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
                                route_credit=dict(none='factorized race: winner value credit, common first-time clock '
                                                       'credit only', linear='factorized race plus linearized '
                                                       'local-expectation score credit pi_i g.(v_i - v_bar)',
                                                  linear_rw='factorized race plus linearized local-expectation score credit '
                                                  'for the read value and the write slot')[a.route_credit],
                                kernels='compiled layer steps (torch.compile/inductor, contract-tested against the batched path)'
                                if a.compiled else 'eager batched path'))
    window_times = []
    checkpoint = out.parent / 'checkpoints' / f'{a.tag}.pt'
    first, prior_wall = 0, 0.
    if a.resume:
        state = torch.load(checkpoint, weights_only=False)
        if state['args'] != {k: v for k, v in vars(a).items() if k != 'resume'}:
            raise ValueError('Checkpoint arguments differ')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        if schedule is not None:
            schedule.load_state_dict(state['schedule'])
        rng.bit_generator.state = state['rng']; torch.set_rng_state(state['torch_rng'])
        first, seen, curve, ledger, traced_chars = state['window'], state['seen'], state['curve'], state['ledger'], state['traced_chars']
        window_times, prior_wall = state['window_times'], state['wall_s']
        result['resumed_from_window'] = result.get('resumed_from_window', []) + state.get('resumed_from_window', []) + [first]
    elif checkpoint.exists():
        raise ValueError('Checkpoint exists: explicit --resume required')
    for w in range(first, total_windows):
        starts = rng.integers(0, len(fit) - S - 1, B)
        rows = rows_of(fit, starts, S); y = torch.tensor(np.stack([fit[s + 1:s + S + 1] for s in starts])).long()
        seed = 100000 + a.seed * 1000 + w
        box = {}
        def step(logits=None):
            model.train(); opt.zero_grad(set_to_none=True)
            z = (logits or LOGITS)(model, rows, seed, all_logits=True,
                                   route_credit=None if a.route_credit == 'none' else a.route_credit)
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
        if a.checkpoint_every and (w + 1) % a.checkpoint_every == 0 and w + 1 < total_windows:
            checkpoint.parent.mkdir(exist_ok=True); tmp = checkpoint.with_suffix('.tmp')
            torch.save(dict(args={k: v for k, v in vars(a).items() if k != 'resume'}, window=w + 1, seen=seen,
                            curve=curve, ledger=ledger, traced_chars=traced_chars, window_times=window_times,
                            wall_s=prior_wall + time.perf_counter() - started, model=model.state_dict(),
                            optimizer=opt.state_dict(), schedule=schedule.state_dict() if schedule is not None else None,
                            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                            resumed_from_window=result.get('resumed_from_window', [])), tmp)
            os.replace(tmp, checkpoint)
    work_per_char = (ledger['arithmetic_flops'] + ledger['special_function_evaluations']) / traced_chars if traced_chars else None
    result.update(curve=curve, fitting_chars=seen, windows=total_windows,
                  train_chars_per_s=float(S * B / np.mean(window_times[a.trace_windows:] or window_times)),
                  work=dict(fit_unit_special_flops_per_char_estimate=work_per_char,
                            whole_fit_unit_special_flops_estimate=work_per_char * seen if work_per_char else None,
                            scope='first windows fully traced, extrapolated per character; evaluation separate'))
    if not a.max_windows:
        result['dev_bpc'], result['dev_targets'] = window_scores(model, dev, S, 314159, B)
        test = load_text(95_000_000, a.test)
        result['test_bpc'], result['test_targets'] = window_scores(model, test, S, 314159, B)
        if a.eval_segment and a.eval_segment != S:      # same weights, E64 windows of another length (e.g. T=256)
            E = a.eval_segment; lanes = max(1, B * S // E)
            result['eval_segment'] = E
            result['dev_bpc_eval_segment'], _ = window_scores(model, dev, E, 314159, lanes)
            result['test_bpc_eval_segment'], result['test_targets_eval_segment'] = window_scores(model, test, E, 314159, lanes)
    weights = out.parent / 'checkpoints' / f'{a.tag}_final.pt'      # final weights for inference/continuation analyses
    weights.parent.mkdir(exist_ok=True); torch.save(model.state_dict(), weights)
    result['final_weights'] = str(weights.relative_to(ROOT))
    result.update(status='completed', wall_s=prior_wall + time.perf_counter() - started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('dev_bpc', 'test_bpc', 'train_chars_per_s', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
