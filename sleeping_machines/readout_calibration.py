"""Fit-only readout conditioning, including support of static metadata."""
import torch


@torch.no_grad()
def condition_readout(model, summaries, project_constant_count=True):
    x = summaries.double()
    center = x.mean(0)
    raw_scale = x.std(0, unbiased=False)
    scale = raw_scale.clamp_min(1e-4)
    z = (x-center)/scale
    eig, vectors = torch.linalg.eigh(z.T@z/len(z))
    whitener = (vectors*(eig.clamp_min(0)+.1).rsqrt()[None])@vectors.T
    # The final coordinate is log input count, independent of learned weights.
    # If count never varies in training, its response cannot be identified.
    # Drop its input contribution instead of magnifying it at a new length.
    # Hidden coordinates can change as the core learns; do not project those
    # merely because their initial variance was small.
    unsupported = bool(project_constant_count and raw_scale[-1] <= 1e-12)
    if unsupported:
        whitener[-1, :] = 0
    model.center.copy_(center)
    model.scale.copy_(scale)
    model.whitener.copy_(whitener)
    return {"fit_only": True, "fixed_after_initialization": True, "shrinkage": .1,
            "constant_count_projected": unsupported,
            "effective_rank": float(eig.sum().square()/eig.square().sum().clamp_min(1e-20))}
