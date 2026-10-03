"""Single-site branch-content averaging with correctly scaled route credit.

The native inference program is unchanged. This training-only helper does not
claim a complete expected-risk derivative or lower shared-noise batch variance.
"""
import torch
from torch.nn import functional as F
from .batched_episodes import batched_logits


def conditional_branch_objective(model,rows,seed,selected_races,content=True):
    """Return a normalized joint or matched choice-only objective and evidence.

    The caller chooses one uniform legal race per episode. Content averaging
    replaces the factual loss derivative; only categorical credit is R-scaled.
    Factual pre-update predictions never read targets or selected alternatives.
    """
    if not rows or len(selected_races)!=len(rows):raise ValueError('One selected race per nonempty episode required')
    lengths=[len(row['events'])*model.depth*model.heads for row in rows]
    for race,R in zip(selected_races,lengths):
        if isinstance(race,bool) or not isinstance(race,int) or not 0<=race<R:
            raise ValueError('Selected race outside legal episode')
    scores=[]
    factual=batched_logits(model,rows,seed,record=scores)
    selected=torch.stack([scores[race][j] for j,race in enumerate(selected_races)])
    pi=selected.softmax(-1)
    lanes=[row for row in rows for _ in range(model.pool)]
    forces=[(race,alternative) for race in selected_races for alternative in range(model.pool)]
    if content:
        shadow=batched_logits(model,lanes,seed,forces)
    else:
        with torch.no_grad():shadow=batched_logits(model,lanes,seed,forces)
    losses=F.cross_entropy(shadow,torch.tensor([row['target'] for row in lanes]),reduction='none').view(len(rows),model.pool)
    factual_losses=F.cross_entropy(factual,torch.tensor([row['target'] for row in rows]),reduction='none')
    scale=pi.new_tensor(lengths)
    choice=(scale*(pi*losses.detach()).sum(-1)).mean()
    branch=(pi.detach()*losses).sum(-1).mean() if content else factual_losses.mean()
    return dict(objective=branch+choice,branch_objective=branch,choice_objective=choice,
        factual_logits=factual,factual_losses=factual_losses,branch_losses=losses,probabilities=pi,
        selected_races=list(selected_races),legal_races=lengths,
        shadow_lanes=len(lanes),shadow_events=sum(len(row['events']) for row in lanes),
        content_credit='conditional live branch average' if content else 'factual winner only')
