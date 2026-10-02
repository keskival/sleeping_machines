"""Bounded actual-state suffix credit, retaining native sparse inference."""
from contextlib import contextmanager
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
import dvs_native_benchmark as N
import sleeping_machines.batched_addressed_fit as K
from dvs_counterfactual_route_audit import ConditionalRoute

BASE_SOURCES = N.sources
REFERENCE_ROUTE = K.BatchedTemporalRoute


def sources():
    names = ['experiments/dvs_state_clock_credit_benchmark.py',
        'experiments/dvs_state_clock_credit_contracts.py',
        'experiments/dvs_counterfactual_route_audit.py',
        'sleeping_machines/batched_addressed_fit.py',
        'experiments/theory/81_bounded_state_clock_credit.md']
    return {**BASE_SOURCES(), **{name:N.sha(ROOT/name) for name in names}}


def joint_credit(scores, first, losses, baseline):
    """Conditional winner mean of the joint winner/time likelihood score."""
    rates = scores.double().exp(); pi = rates/rates.sum(-1, keepdim=True)
    centered = losses.double()-baseline.double()[:, None]
    return pi*centered-rates*first[:, None]*(pi*centered).sum(-1, keepdim=True)


@contextmanager
def trace_site(site, head, alternative=None):
    original = K.BatchedTemporalRoute; records = []; counter = 0
    def apply(scores, values, noise):
        nonlocal counter
        selected = counter == site; counter += 1
        if selected:
            with torch.no_grad():
                first, winner = (noise[None]/scores.detach().double().exp()).min(-1)
            records.append(dict(scores=scores, values=values, first=first[:, head],
                winner=winner[:, head], head=head))
        if selected and alternative is not None:
            forced = torch.full(scores.shape[:2], -1, dtype=torch.long, device=scores.device)
            forced[:, head] = alternative
            return ConditionalRoute.apply(scores, values, noise, forced, forced)
        return REFERENCE_ROUTE.apply(scores, values, noise)
    K.BatchedTemporalRoute = SimpleNamespace(apply=apply)
    try:
        yield records
        if len(records) != 1:raise ValueError('Exactly one eligible corrected race required')
    finally:
        K.BatchedTemporalRoute = original


def corrected_loss(model, rows, seed, epoch):
    if model.pool != 2:raise ValueError('Two-candidate bounded replay required')
    head = (epoch-1)%model.heads; depth = ((epoch-1)//model.heads)%model.depth
    site = 9*model.depth+depth
    targets = torch.tensor([row['target'] for row in rows], device=model.embedding.weight.device)
    with trace_site(site, head) as records:
        logits, state, _ = K.forward(model, rows, seed)
    record = records[0]
    factual = F.cross_entropy(logits, targets, reduction='none')
    with torch.no_grad():
        # The baseline depends only on candidates constructed before this draw.
        baseline = F.cross_entropy(model.head(record['values'].mean(2).reshape(len(rows), -1)),
            targets, reduction='none')
        with trace_site(site, head, 1-record['winner']) as shadows:
            alternative, _, _ = K.forward(model, rows, seed)
        alternative_loss = F.cross_entropy(alternative, targets, reduction='none')
        losses = torch.where(record['winner'][:, None] == torch.arange(2, device=targets.device),
            factual.detach()[:, None], alternative_loss[:, None])
        credit = joint_credit(record['scores'][:, head], record['first'], losses, baseline)
    def replace(gradient):
        result = gradient.clone(); result[:, head] = credit.to(result.dtype); return result
    record['scores'].register_hook(replace)
    return factual.sum(), state, dict(record=record, shadow=shadows[0], credit=credit,
        outcome_losses=losses, baseline=baseline, logits=logits, site=site, head=head)


def train_window(model, optimizer, rows, a, epoch, trace=False):
    optimizer.zero_grad(set_to_none=True); stages = {}; box = {}
    def traced(name, fn):
        if trace:stages.setdefault(name, []).append(N.capture(fn))
        else:fn()
    def compute():
        loss, state, _ = corrected_loss(model, rows, 100000+a.seed+10000*epoch, epoch)
        box.update(loss=loss, state=state)
    traced('forward_and_loss', compute)
    if not torch.isfinite(box['loss']):raise FloatingPointError('Nonfinite actual-state loss')
    traced('backward', lambda:box['loss'].backward())
    def normalize():
        for parameter in model.parameters():
            if parameter.grad is not None:parameter.grad.div_(len(rows))
    traced('gradient_normalization', normalize)
    traced('gradient_clipping', lambda:torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    traced('optimizer', optimizer.step)
    # Include the full shadow prefix and suffix in actual simulated fitting activity.
    events = 2*box['state']['events']; selected = events*model.depth*model.heads
    return dict(targets=len(rows), loss_sum=float(box['loss'].detach()), events=events,
        key_scores=selected*model.pool, selected_updates=selected,
        counterfactual_values=selected*model.pool,
        stages={name:N.merge(values) for name,values in stages.items()})


@contextmanager
def activate():
    previous = N.sources, N.train_window
    N.sources, N.train_window = sources, train_window
    try:yield
    finally:N.sources, N.train_window = previous


def run(a, directory=None):
    with activate():result = N.run(a, directory)
    if result['status'] == 'completed':
        result['credit_protocol'] = dict(event=9, corrected_heads_per_clip_per_window=1,
            rotation='Head then layer across fixed epochs', full_shadow_forwards_per_window=1,
            baseline='Detached existing class head on pre-draw mean candidate messages',
            gradient='Replace one node score gradient by conditional-winner joint winner/time likelihood credit',
            activity_scope='Factual plus complete alternative forward state updates, key scores and candidate values; no extra supervised targets',
            inference_architecture_unchanged=True, full_model_gradient_exact=False)
        path = Path(directory) if directory else ROOT/'experiments/results/dvs_native'
        (path/(a.tag+'.json')).write_text(__import__('json').dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':run(N.parser().parse_args())
