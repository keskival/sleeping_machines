"""Unadmitted compact suffix objective; preserve factual backward and all choices."""
import torch
from torch.nn import functional as F
from causal_language_replay_helpers_rng import row
from sleeping_machines.causal_language_shadow_cached_prefix import batched_chunks
from sleeping_machines.causal_language_shadow_compact_suffix import compact_suffix_returns


def batched_objective(model, tokens, targets, entering, seed):
    model.train()
    scores, winners, snapshots = [], [], []
    observed = row(model, tokens, entering)
    logits, states, end_rng = batched_chunks(model, [observed], seed, [entering],
        record=scores, winners=winners, snapshots=snapshots)
    losses = F.cross_entropy(logits[0], targets, reduction='none')
    events = [r // (model.depth * model.heads) for r in range(len(scores))]
    forces = [(r, i) for r in range(len(scores)) for i in range(model.pool)
              if i != int(winners[r][0])]
    with torch.no_grad():
        suffix = torch.stack([losses.detach()[t:].sum() for t in range(len(tokens))])
        returns = suffix[events, None].expand(len(scores), model.pool).clone()
        if forces:
            alternative, activity = compact_suffix_returns(model, observed, targets, snapshots, forces)
            assert torch.equal(activity['factual_end_rng'], end_rng)
            for j, (r, i) in enumerate(forces):
                returns[r, i] = alternative[j]
        else:
            activity = dict(shadow_lanes=0, shadow_events=0, token_loop_iterations=0)
    objective = losses.sum() + sum((s[0].softmax(0) * u).sum() for s, u in zip(scores, returns))
    activity.update(factual_loss_sum=float(losses.detach().sum()), factual_end_rng=end_rng,
                    reused_winner_returns=len(scores))
    return objective, logits[0], states[0], activity
