"""All-route full-suffix write credit reusing causal factual token prefixes."""
import torch
from torch.nn import functional as F
from causal_language_replay_helpers_rng import row
from sleeping_machines.causal_language_shadow_cached_prefix import batched_chunks


def batched_objective(model,tokens,targets,entering,seed):
    model.train();scores=[];winners=[];snapshots=[];observed=row(model,tokens,entering)
    logits,states,end_rng=batched_chunks(model,[observed],seed,[entering],record=scores,winners=winners,snapshots=snapshots)
    losses=F.cross_entropy(logits[0],targets,reduction='none');T=len(tokens);R=len(scores);U=model.pool
    width=model.depth*model.heads;shadow_lanes=0;shadow_events=0;groups=0
    with torch.no_grad():
        suffix=torch.stack([losses.detach()[k:].sum() for k in range(T)])
        returns=suffix.repeat_interleave(width)[:,None].expand(R,U).clone()
        for event,frame in enumerate(snapshots):
            forces=[(local,i) for local in range(width) for i in range(U)
                    if i!=int(winners[event*width+local][0])]
            if not forces:continue
            observed_suffix=row(model,tokens[event:],frame['state'])
            alternative,_,_=batched_chunks(model,[observed_suffix]*len(forces),frame['rng'],[frame['state']]*len(forces),forces)
            utilities=F.cross_entropy(alternative.reshape(-1,model.vocabulary),targets[event:].repeat(len(forces)),
                                     reduction='none').reshape(len(forces),T-event).sum(-1)
            for j,(local,i) in enumerate(forces):returns[event*width+local,i]=utilities[j]
            shadow_lanes+=len(forces);shadow_events+=len(forces)*(T-event);groups+=1
    objective=losses.sum()+sum((s[0].softmax(0)*u).sum() for s,u in zip(scores,returns))
    snapshot_bytes=sum(frame['state'].storage()['persistent_tensor_bytes']+frame['rng'].numel()*frame['rng'].element_size()
                       for frame in snapshots)
    return objective,logits[0],states[0],dict(shadow_lanes=shadow_lanes,shadow_events=shadow_events,
        reused_winner_returns=R,shadow_groups=groups,snapshot_tensor_bytes=snapshot_bytes,
        factual_loss_sum=float(losses.detach().sum()),factual_end_rng=end_rng)
