"""Exact conditional winner-return reuse; only losing routes execute shadow lanes."""
import torch
from torch.nn import functional as F
from causal_language_replay_helpers_rng import row
import sleeping_machines.causal_language_shadow_rng as K


def objective(model,tokens,targets,entering,seed):
    model.train();scores=[];winners=[];observed=row(model,tokens,entering)
    original=K.LaneRace
    class RecordedRace(original):
        @staticmethod
        def forward(ctx,*args):
            result=original.forward(ctx,*args);winners.append(int(result[2][0]));return result
    K.LaneRace=RecordedRace
    try:logits,states,end_rng=K.batched_chunks(model,[observed],seed,[entering],record=scores)
    finally:K.LaneRace=original
    losses=F.cross_entropy(logits[0],targets,reduction='none');R=len(scores)
    assert len(winners)==R
    forces=[(r,i) for r in range(R) for i in range(model.pool) if i!=winners[r]]
    with torch.no_grad():
        # Every winner return is already available; detach before adding route credit.
        returns=torch.stack([losses.detach()[r//(model.depth*model.heads):].sum().repeat(model.pool) for r in range(R)])
        if forces:
            alternative,_,_=K.batched_chunks(model,[observed]*len(forces),seed,[entering]*len(forces),forces)
            utilities=F.cross_entropy(alternative.reshape(-1,model.vocabulary),targets.repeat(len(forces)),reduction='none').reshape(len(forces),len(tokens))
            for j,(r,i) in enumerate(forces):returns[r,i]=utilities[j,r//(model.depth*model.heads):].sum()
    result=losses.sum()+sum((s[0].softmax(0)*u).sum() for s,u in zip(scores,returns))
    return result,logits[0],states[0],dict(shadow_lanes=len(forces),shadow_events=len(forces)*len(tokens),factual_loss_sum=float(losses.detach().sum()),factual_end_rng=end_rng,winner_returns_reused=R)
