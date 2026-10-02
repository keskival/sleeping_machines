"""Actual-write choice utility with the native pathwise timing credit retained."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_state_clock_credit_benchmark as S

N, K = S.N, S.K
REFERENCE_ROUTE = S.REFERENCE_ROUTE


def sources():
    names = ['experiments/dvs_state_choice_credit_benchmark.py',
        'experiments/dvs_state_choice_credit_contracts.py',
        'experiments/dvs_state_clock_credit_benchmark.py',
        'experiments/dvs_counterfactual_route_audit.py',
        'sleeping_machines/batched_addressed_fit.py',
        'experiments/theory/84_actual_write_choice_with_timing.md']
    return {**S.BASE_SOURCES(), **{name:N.sha(ROOT/name) for name in names}}


class ChoiceWithTimingRoute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, values, noise, box):
        ctx.box = box
        return REFERENCE_ROUTE.forward(ctx, scores, values, noise)

    @staticmethod
    def backward(ctx, error_value, error_delay, unused):
        gradient, value_gradient, _ = REFERENCE_ROUTE.backward(ctx, error_value, error_delay, unused)
        rates, first, winner, values = ctx.saved_tensors
        box = ctx.box; head = box['head']
        if 'credit' not in box:raise RuntimeError('Conditional utility must precede backward')
        corrected = box['credit'].clone()
        if error_delay is not None:
            timing = -error_delay[:,head]*.010*first[:,head]/(1+first[:,head]).square()
            corrected.scatter_add_(1, winner[:,head,None], timing[:,None])
        gradient[:,head] = corrected.to(gradient.dtype)
        if box.get('audit', False):
            box.update(error_delay=None if error_delay is None else error_delay.detach(),
                corrected_score_gradient=gradient.detach(), value_gradient=value_gradient.detach())
        return gradient, value_gradient, None, None


@contextmanager
def trace_site(site, head, audit=False):
    original = K.BatchedTemporalRoute; records = []; counter = 0
    def apply(scores, values, noise):
        nonlocal counter
        selected = counter == site; counter += 1
        if selected:
            with torch.no_grad():
                first, winner = (noise[None]/scores.detach().double().exp()).min(-1)
            box = dict(scores=scores, values=values, first=first[:,head],
                winner=winner[:,head], head=head, audit=audit)
            records.append(box)
            return ChoiceWithTimingRoute.apply(scores, values, noise, box)
        return REFERENCE_ROUTE.apply(scores, values, noise)
    K.BatchedTemporalRoute = SimpleNamespace(apply=apply)
    try:
        yield records
        if len(records) != 1:raise ValueError('Exactly one eligible corrected race required')
    finally:K.BatchedTemporalRoute = original


def corrected_loss(model, rows, seed, epoch, audit=False):
    if model.pool != 2:raise ValueError('Two-candidate actual-write comparison required')
    head = (epoch-1)%model.heads; depth = ((epoch-1)//model.heads)%model.depth
    site = 9*model.depth+depth
    targets = torch.tensor([row['target'] for row in rows], device=model.embedding.weight.device)
    with trace_site(site, head, audit) as records:
        logits, state, _ = K.forward(model, rows, seed)
    box = records[0]; factual = F.cross_entropy(logits, targets, reduction='none')
    with torch.no_grad():
        with S.trace_site(site, head, 1-box['winner']) as shadows:
            alternative, shadow_state, _ = K.forward(model, rows, seed)
        alternative_loss = F.cross_entropy(alternative, targets, reduction='none')
        losses = torch.where(box['winner'][:,None] == torch.arange(2, device=targets.device),
            factual.detach()[:,None], alternative_loss[:,None])
        probabilities = box['scores'][:,head].double().softmax(-1)
        box['credit'] = probabilities*(losses.double()-(probabilities*losses.double()).sum(-1,keepdim=True))
    return factual.sum(), state, dict(record=box, shadow=shadows[0], shadow_state=shadow_state,
        logits=logits, outcome_losses=losses, site=site, head=head)


@contextmanager
def activate():
    previous = N.sources, N.train_window, S.corrected_loss
    N.sources, N.train_window, S.corrected_loss = sources, S.train_window, corrected_loss
    try:yield
    finally:N.sources, N.train_window, S.corrected_loss = previous


def run(a, directory=None):
    with activate():result = N.run(a, directory)
    if result['status'] == 'completed':
        result['credit_protocol'] = dict(kind='state_choice',event=9,corrected_heads_per_clip_per_window=1,
            rotation='Head then layer across fixed epochs',full_shadow_forwards_per_window=1,
            gradient='Exact conditional actual-write choice utility plus unchanged native pathwise timing score credit',
            timing_gradient_retained=True,baseline=None,inference_architecture_unchanged=True,
            activity_scope='Factual and full alternative replay keys, writes and candidate values; one supervised target per clip',
            full_model_gradient_exact=False, downstream_timing_jumps_corrected=False)
        path = Path(directory) if directory else ROOT/'experiments/results/dvs_native'
        (path/(a.tag+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':run(N.parser().parse_args())
