"""Reuse factual selected-write returns; enumerate only different outcomes."""
import torch
from torch.nn import functional as F
from causal_language_replay_helpers_rng import row
from sleeping_machines.causal_language_shadow_winner_reuse import batched_chunks


def batched_objective(model,tokens,targets,entering,seed):
    model.train();scores=[];winners=[];observed=row(model,tokens,entering)
    logits,states,end_rng=batched_chunks(model,[observed],seed,[entering],record=scores,winners=winners)
    losses=F.cross_entropy(logits[0],targets,reduction='none');R=len(scores);U=model.pool
    events=[r//(model.depth*model.heads) for r in range(R)]
    with torch.no_grad():
        # Match the frozen return reduction and its complete operator accounting.
        suffix=torch.stack([losses.detach()[event:].sum() for event in range(len(tokens))])
        returns=suffix[events,None].expand(R,U).clone()
        forces=[(race,i) for race in range(R) for i in range(U) if i!=int(winners[race][0])]
        if forces:
            alternative,_,_=batched_chunks(model,[observed]*len(forces),seed,[entering]*len(forces),forces)
            utilities=F.cross_entropy(alternative.reshape(-1,model.vocabulary),targets.repeat(len(forces)),
                                     reduction='none').reshape(len(forces),len(tokens))
            for j,(race,i) in enumerate(forces):returns[race,i]=utilities[j,events[race]:].sum()
    objective=losses.sum()+sum((s[0].softmax(0)*u).sum() for s,u in zip(scores,returns))
    return objective,logits[0],states[0],dict(shadow_lanes=len(forces),shadow_events=len(forces)*len(tokens),
        reused_winner_returns=R,factual_loss_sum=float(losses.detach().sum()),factual_end_rng=end_rng)
