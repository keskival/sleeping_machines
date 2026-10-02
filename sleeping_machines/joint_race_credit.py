"""Detached LOCAL joint winner/time credit reference; no model installs this.

Branch losses must include actual addressed-state/suffix consequences. Affine
surrogate coefficients and proposal probabilities must be prefix-conditioned,
independent of the current race draw. Direct prediction derivatives are separate.
"""
import torch


def _inputs(scores, time, intercept, slope):
    scores = scores.detach().to(torch.float64)
    intercept = intercept.detach().to(scores)
    slope = slope.detach().to(scores)
    time = torch.as_tensor(time, dtype=scores.dtype, device=scores.device).detach()
    if scores.ndim != 1 or not scores.numel() or intercept.shape != scores.shape or slope.shape != scores.shape:
        raise ValueError('Matching nonempty score/coefficient vectors required')
    if time.ndim or not bool(torch.isfinite(time)) or float(time) < 0:
        raise ValueError('Finite nonnegative scalar race time required')
    if not all(bool(torch.isfinite(x).all()) for x in (scores, intercept, slope)):
        raise ValueError('Finite scores and coefficients required')
    rates = scores.exp()
    total = rates.sum()
    if not bool(torch.isfinite(total)) or not bool((rates > 0).all()):
        raise ValueError('Finite total and strictly positive candidate rates required')
    p = rates / total
    analytic = p * (intercept - (p * intercept).sum()
                    + (slope - 2 * (p * slope).sum()) / total)
    return rates, p, time, intercept, slope, analytic


def enumerated_credit(scores, time, branch_losses, intercept, slope):
    """Analytic affine mean plus winner-enumerated residual at sampled time.

Exact local expected credit after averaging time, if supplied losses are the
actual conditional branch losses and independence/integrability conditions hold.
This vector is not the expected sequence gradient at one sampled time.
"""
    rates, p, time, intercept, slope, analytic = _inputs(scores, time, intercept, slope)
    losses = branch_losses.detach().to(p)
    if losses.shape != p.shape or not bool(torch.isfinite(losses).all()):
        raise ValueError('One finite actual suffix loss per candidate required')
    residual = losses - intercept - slope * time
    return analytic + p * residual - rates * time * (p * residual).sum()


def sampled_credit(scores, time, branch_loss, candidate, proposal, intercept, slope):
    """One importance-weighted candidate residual; proposal must have full support."""
    rates, p, time, intercept, slope, analytic = _inputs(scores, time, intercept, slope)
    q = proposal.detach().to(p)
    loss = torch.as_tensor(branch_loss, dtype=p.dtype, device=p.device).detach()
    if q.shape != p.shape or not bool(torch.isfinite(q).all()) or not bool((q > 0).all()):
        raise ValueError('Finite full-support proposal required')
    if not bool(torch.isclose(q.sum(), q.new_tensor(1.), atol=1e-10, rtol=1e-10)):
        raise ValueError('Normalized proposal required')
    if not isinstance(candidate, int) or not 0 <= candidate < p.numel():
        raise ValueError('Valid integer candidate required')
    if loss.ndim or not bool(torch.isfinite(loss)):
        raise ValueError('Finite scalar actual suffix loss required')
    score = -rates * time
    score[candidate] += 1
    residual = loss - intercept[candidate] - slope[candidate] * time
    return analytic + (p[candidate] / q[candidate]) * score * residual
