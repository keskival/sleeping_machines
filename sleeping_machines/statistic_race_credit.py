"""Statistic-valued LOCAL delivery and one-next-event write credit contracts.

No integrated model installs these primitives. Current counts are fixed for
delivery credit; next-visit distribution must be supplied for expected write
credit. Candidate discovery, all lookups and fitting that distribution cost work.
"""
import torch


def predictive(counts, alpha=.5):
    counts=counts.detach().to(torch.float64)
    if counts.ndim not in (1,2) or counts.shape[-1]<1 or not bool(torch.isfinite(counts).all()) or not bool((counts>=0).all()):
        raise ValueError('Finite nonnegative nonempty count vectors required')
    if not isinstance(alpha,(float,int)) or not 0<float(alpha)<float('inf'):
        raise ValueError('Finite positive symmetric pseudocount required')
    return (counts+alpha)/(counts.sum(-1,keepdim=True)+counts.shape[-1]*alpha)


def delivery_credit(scores, counts, target, alpha=.5):
    """Exact current expected hard-route loss gradient, not mixture NLL gradient."""
    probabilities=predictive(counts,alpha)
    scores=scores.detach().to(probabilities)
    if probabilities.ndim!=2 or scores.ndim!=1 or scores.shape[0]!=probabilities.shape[0] or not bool(torch.isfinite(scores).all()):
        raise ValueError('One finite score per candidate required')
    if not isinstance(target,int) or not 0<=target<probabilities.shape[1]:
        raise ValueError('Valid target symbol required')
    losses=-probabilities[:,target].log()
    p=scores.softmax(0)
    return p*(losses-(p*losses).sum())


def expected_write_gain(counts, target, future_distribution, alpha=.5):
    """Expected next-event LOG predictive improvement for one symbol increment.

Future law q is fixed and supplied, not recovered from present counts. This is
not the whole-sequence write benefit or a neural addressed-state gradient.
"""
    predictive(counts,alpha)  # shared count/pseudocount validation
    counts=counts.detach().to(torch.float64)
    q=future_distribution.detach().to(counts)
    if counts.ndim!=1 or q.shape!=counts.shape or not bool(torch.isfinite(q).all()) or not bool((q>=0).all()) or not bool(torch.isclose(q.sum(),q.new_tensor(1.),atol=1e-10,rtol=1e-10)):
        raise ValueError('One normalized finite future law required')
    if not isinstance(target,int) or not 0<=target<counts.numel():
        raise ValueError('Valid written symbol required')
    return q[target]*torch.log1p(1/(counts[target]+alpha))-torch.log1p(1/(counts.sum()+counts.numel()*alpha))
