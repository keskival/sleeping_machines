"""Episode-batched DVS training with optional all-race replay credit through batched shadow lanes (THEORY §405).

Window step: one batched factual pass with gradients over the window's episodes (sleeping_machines/batched_episodes;
factorized race law), recording every race's scores.  With --route-credit, every alternative of every race of every
episode is replayed in one batched, gradient-free shadow pass (first-time-preserving forces, common random numbers),
and the route surrogate sum_j sum_r sum_i softmax(s_{j,r})_i * stopgrad(L_{j,r}(i)) is added: exact local-expectation
credit over all races (no race sampling).  Evaluation is batched too (same noise seed and outputs as the sequential
evaluation).  Equivalence with the sequential drivers is contract-tested; work is traced on sampled windows.
"""
from contextlib import contextmanager
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_local_expectation_benchmark as LE  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402
from parallel_head_accumulated_language import merge  # noqa: E402
from race_language_screen import capture  # noqa: E402
from sleeping_machines.batched_episodes import batched_logits  # noqa: E402

REPLAYS = [0]
COMPILED, TRACING = [False], [False]


def factual_logits(model, rows, seed, record=None):
    """--compiled (§412): untraced factual passes without recorded scores use compiled layer steps
    (sleeping_machines/compiled_episodes; contract-tested against batched_logits).  Route credit, shadow lanes and
    traced windows stay on the batched path, whose operators the work tracer sees."""
    if COMPILED[0] and record is None and not TRACING[0]:
        from sleeping_machines.compiled_episodes import compiled_logits
        return compiled_logits(model, rows, seed)
    return batched_logits(model, rows, seed, record=record)


def parser():
    p = C.parser(); p.add_argument('--route-credit', action='store_true', help='all-race local-expectation replay credit')
    p.add_argument('--route-races', type=int, default=0, help='sample this many races per episode (0 = all), scaled R/k')
    p.add_argument('--compiled', action='store_true', help='compiled layer steps for untraced factual passes (§412)')
    return p


def route_term(model, rows, seed, scores, k=0, rng=None):
    """exact local-expectation credit over all races (k = 0) or k sampled races per episode (scaled R/k), via one
    batched shadow pass."""
    D, H = model.depth, model.heads
    lanes, forces, index, scale = [], [], [], {}
    for j, r in enumerate(rows):
        races = len(r['events']) * D * H
        chosen = range(races) if not k or k >= races else sorted(rng.choice(races, size=k, replace=False).tolist())
        scale[j] = races / len(chosen)
        for race in chosen:
            for i in range(model.pool):
                lanes.append(r); forces.append((race, i)); index.append((j, race, i))
    with torch.no_grad():
        logits = batched_logits(model, lanes, seed, forces)
        losses = F.cross_entropy(logits, torch.tensor([r['target'] for r in lanes]), reduction='none').double()
    REPLAYS[0] += len(lanes)
    L = {}
    for (j, race, i), loss in zip(index, losses):
        L.setdefault((j, race), torch.zeros(model.pool, dtype=torch.float64))[i] = loss
    total = scores[0].new_zeros(())
    for (j, race), values in L.items():
        total = total + scale[j] * (torch.softmax(scores[race][j], 0) * values.to(scores[race].dtype)).sum()
    return total, len(lanes)


def train_window(model, optimizer, rows, a, epoch, trace=False):
    TRACING[0] = trace
    optimizer.zero_grad(set_to_none=True); stages = {}; box = {}
    seed = 100000 + a.seed + 10000 * epoch
    def traced(name, fn):
        if trace: stages.setdefault(name, []).append(capture(fn))
        else: fn()
    def forward():
        model.train(); scores = []
        if hasattr(model, '_fast_layers'): model._fast_layers = {}
        logits = factual_logits(model, rows, seed, record=scores if a.route_credit else None)
        losses = F.cross_entropy(logits, torch.tensor([r['target'] for r in rows]), reduction='none')
        objective = losses.sum(); replays = 0
        if a.route_credit:
            rng = np.random.default_rng(seed * 7919 + rows[0]['index'])
            route, replays = route_term(model, rows, seed, scores, getattr(a, 'route_races', 0), rng); objective = objective + route
        box.update(loss_sum=float(losses.detach().sum()), objective=objective, replays=replays)
    traced('forward_and_loss', forward)
    if not torch.isfinite(box['objective']): raise FloatingPointError('Nonfinite objective')
    traced('backward', lambda: box['objective'].backward())
    def normalize():
        for parameter in model.parameters():
            if parameter.grad is not None: parameter.grad.div_(len(rows))
    traced('gradient_normalization', normalize)
    traced('gradient_clipping', lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    traced('optimizer', optimizer.step)
    events = sum(len(r['events']) for r in rows); races = events * model.depth * model.heads
    return dict(targets=len(rows), loss_sum=box['loss_sum'], events=events, key_scores=races * model.pool,
                selected_updates=races, counterfactual_values=0, route_replays=box['replays'],
                stages={name: merge(values) for name, values in stages.items()})


@torch.no_grad()
def evaluate(model, rows):
    model.eval()
    TRACING[0] = False
    logits = factual_logits(model, rows, 314159)
    targets = torch.tensor([r['target'] for r in rows])
    losses = F.cross_entropy(logits, targets, reduction='none')
    events = sum(len(r['events']) for r in rows); races = events * model.depth * model.heads
    return dict(targets=len(rows), nll=float(losses.mean()), accuracy=float((logits.argmax(-1) == targets).float().mean()),
                per_target_nll=losses.tolist(), probabilities=logits.softmax(-1).tolist(), events=events,
                key_scores=races * model.pool, selected_updates=races, max_state_tensor_bytes=None)


def configure_compiled(a):
    if getattr(a, 'compiled', False):
        from torch._dynamo import config as dynamo_config
        from torch._inductor import config as inductor_config
        COMPILED[0] = True
        inductor_config.compile_threads = 1          # no compile-worker pool (memory floor)
        dynamo_config.cache_size_limit = 64          # lane counts, grad modes and per-layer gains each specialize


def make_model(a, fast=True):
    configure_compiled(a)
    return C.make_model(a, True)          # batched path needs the fast mixin's stacked parameters


def sources():
    names = ['experiments/dvs_batched_le_benchmark.py', 'sleeping_machines/batched_episodes.py',
             'sleeping_machines/compiled_episodes.py']
    return {**C.sources(), **{n: N.sha(ROOT / n) for n in names}}


@contextmanager
def activate():
    old = N.make_model, N.parser, N.sources, N.train_window, N.evaluate
    N.make_model, N.parser, N.sources, N.train_window, N.evaluate = make_model, parser, sources, train_window, evaluate
    try:
        yield
    finally:
        N.make_model, N.parser, N.sources, N.train_window, N.evaluate = old


if __name__ == '__main__':
    import json
    args = parser().parse_args()
    with activate():
        N.run(args)
    out = ROOT / 'experiments/results/dvs_native' / (args.tag + '.json')
    result = json.loads(out.read_text())
    if result.get('status') == 'completed':
        result['route_credit'] = dict(enabled=args.route_credit, estimator='all-race local expectation via batched shadow lanes'
                                      if args.route_credit else None, replay_lanes_this_process=REPLAYS[0],
                                      replay_work='inside traced forward stages, so included in the work estimate')
        out.write_text(json.dumps(result, indent=2) + '\n')
