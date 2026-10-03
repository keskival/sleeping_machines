"""Diagnostic-only detached return centering; no active training source imports this."""
import torch
from torch.nn import functional as F
from causal_language_replay_helpers_rng import row
from sleeping_machines.causal_language_shadow_winner_reuse import batched_chunks


class CenteredProgram:
    def __init__(self, reuse):
        self.reuse = reuse
        self.batched_chunks = batched_chunks

    def batched_objective(self, model, tokens, targets, entering, seed):
        model.train()
        scores, winners = [], []
        observed = row(model, tokens, entering)
        logits, states, end_rng = self.batched_chunks(
            model, [observed], seed, [entering], record=scores, winners=winners)
        losses = F.cross_entropy(logits[0], targets, reduction='none')
        races, pool = len(scores), model.pool
        events = [r // (model.depth * model.heads) for r in range(races)]
        with torch.no_grad():
            suffix = torch.stack([losses.detach()[event:].sum() for event in range(len(tokens))])
            returns = suffix[events, None].expand(races, pool).clone()
            forces = [(r, i) for r in range(races) for i in range(pool)
                      if not self.reuse or i != int(winners[r][0])]
            if forces:
                alternate, _, _ = self.batched_chunks(
                    model, [observed] * len(forces), seed, [entering] * len(forces), forces)
                utilities = F.cross_entropy(alternate.reshape(-1, model.vocabulary),
                    targets.repeat(len(forces)), reduction='none').reshape(len(forces), len(tokens))
                for j, (r, i) in enumerate(forces):
                    returns[r, i] = utilities[j, events[r]:].sum()
            # The winner baseline is read from THIS program's return table;
            # full replay must not substitute factual-lane rounding for its winner.
            centered = torch.stack([u - u[int(winners[r][0])] for r, u in enumerate(returns)])
            assert not centered.requires_grad
        objective = losses.sum() + sum((s[0].softmax(0) * u).sum()
                                      for s, u in zip(scores, centered))
        return objective, logits[0], states[0], dict(
            shadow_lanes=len(forces), shadow_events=len(forces) * len(tokens),
            factual_loss_sum=float(losses.detach().sum()), factual_end_rng=end_rng)
