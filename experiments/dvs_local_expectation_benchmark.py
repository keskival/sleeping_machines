"""DVS native fit with local-expectation counterfactual race credit (THEORY §402).

Wraps the clock-calibrated driver and replaces its window step:
  * the realized episode runs with credit='pathwise' (the winner's exact value and interior delay derivatives;
    no linearized route teacher);
  * k races per episode are sampled uniformly; for each, every alternative i is replayed without gradients with all
    other race noise fixed (common random numbers), giving the episode loss L_r(i);
  * route credit comes from the surrogate (R/k) * sum_r sum_i softmax(s_r)_i * stopgrad(L_r(i)), whose gradient is the
    exact marginal over that race's winner.
The forward, data, noise seeds, optimizer windows, normalization, clipping and Adam are those of the base driver.
Replay forwards are executed and counted in the activity ledger (route_replays); traced windows charge them too.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_clock_calibrated_benchmark as C  # noqa: E402
import dvs_native_benchmark as N  # noqa: E402
from parallel_head_accumulated_language import merge  # noqa: E402
from race_language_screen import capture  # noqa: E402
from sleeping_machines.parallel_head_race_language import ParallelHeadRaceLanguageModel  # noqa: E402

PATHWISE = None  # bound to AddressedEventHeads.race after model construction


def parser():
    p = C.parser(); p.add_argument('--route-samples', type=int, default=4, help='races sampled per episode (k)')
    return p


def run(model, row, seed, force=None, record=None):
    """one episode with fixed noise; force = (race, alternative); record collects scores of every race."""
    counter = [0]

    def race(scores, values=None):
        r = counter[0]; counter[0] += 1
        if force is not None and r == force[0]:
            rates = scores.to(torch.float64).exp()
            times = torch.empty_like(rates).exponential_() / rates          # identical RNG consumption
            i = force[1]; t = times[i]
            return values[i], .001 + .010 * t / (1 + t), torch.tensor(i)
        if record is not None:
            record.append(scores)
        return PATHWISE(model, scores, values)
    model.race = race
    try:
        state = model.new_state()
        if model.training and hasattr(model, '_fast_layers'):
            model._fast_layers = {}
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for timestamp, content in row['events']:
                logits, _ = model.consume_event(0, timestamp, content, state)
    finally:
        del model.race
        if hasattr(model, '_fast_layers'):
            model._fast_layers = None
    return F.cross_entropy(logits[None], torch.tensor([row['target']])), counter[0], state


REPLAYS = [0]


def train_window(model, optimizer, rows, a, epoch, trace=False):
    optimizer.zero_grad(set_to_none=True); stages = {}; loss_sum = 0.; events = keys = commits = teachers = replays = 0
    def traced(name, fn):
        if trace: stages.setdefault(name, []).append(capture(fn))
        else: fn()
    for row in rows:
        seed = 100000 + a.seed + 10000 * epoch
        rng = np.random.default_rng(seed * 7919 + row['index'])
        box = {}
        def forward():
            model.train(); scores = []
            loss, races, state = run(model, row, seed, record=scores)
            chosen = rng.choice(races, size=min(a.route_samples, races), replace=False)
            route = loss.new_zeros(())
            with torch.no_grad():
                losses = {int(r): torch.tensor([float(run(model, row, seed, force=(int(r), i))[0])
                                                for i in range(len(scores[int(r)]))]) for r in chosen}
            for r, L in losses.items():
                route = route + (races / len(chosen)) * (torch.softmax(scores[r], 0) * L.to(scores[r].dtype)).sum()
            box.update(loss=loss, objective=loss + route, state=state, replays=sum(len(L) for L in losses.values()))
        traced('forward_and_loss', forward)
        if not torch.isfinite(box['loss']): raise FloatingPointError('Nonfinite query loss')
        traced('backward', lambda: box['objective'].backward()); loss_sum += float(box['loss'].detach())
        state = box['state']; events += state.events; keys += state.candidate_scores
        commits += state.selected_updates; teachers += state.counterfactual_values; replays += box['replays']
        REPLAYS[0] += box['replays']
    def normalize():
        for parameter in model.parameters():
            if parameter.grad is not None: parameter.grad.div_(len(rows))
    traced('gradient_normalization', normalize)
    traced('gradient_clipping', lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    traced('optimizer', optimizer.step)
    return dict(targets=len(rows), loss_sum=loss_sum, events=events, key_scores=keys, selected_updates=commits,
                counterfactual_values=teachers, route_replays=replays,
                stages={name: merge(values) for name, values in stages.items()})


def make_model(a, fast=True):
    global PATHWISE
    model = C.make_model(a, fast); model.credit = 'pathwise'
    PATHWISE = type(model).race
    return model


def sources():
    return {**C.sources(), 'experiments/dvs_local_expectation_benchmark.py': N.sha(ROOT / 'experiments/dvs_local_expectation_benchmark.py')}


@contextmanager
def activate():
    old = N.make_model, N.parser, N.sources, N.train_window
    N.make_model, N.parser, N.sources, N.train_window = make_model, parser, sources, train_window
    try:
        yield
    finally:
        N.make_model, N.parser, N.sources, N.train_window = old


if __name__ == '__main__':
    import json
    args = parser().parse_args()
    with activate():
        N.run(args)
    out = ROOT / 'experiments/results/dvs_native' / (args.tag + '.json')
    result = json.loads(out.read_text())
    if result.get('status') == 'completed':   # the base ledger has fixed keys: record the replay count beside it
        result['route_credit'] = dict(estimator='local expectation over sampled races, common random numbers (Theory §402)',
                                      route_samples_per_episode=args.route_samples, replay_forwards_this_process=REPLAYS[0],
                                      replay_work='inside traced forward stages, so included in the work estimate')
        out.write_text(json.dumps(result, indent=2) + '\n')
