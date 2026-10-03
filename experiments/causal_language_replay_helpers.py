"""Independent sequential and batched actual-write all-target language return."""
import copy
import torch
from torch.nn import functional as F
from sleeping_machines.factorized_race import factorized_race,force_at_first_time
from sleeping_machines.causal_language_shadow import batched_chunks


def row(model, tokens, entering):
    return dict(events=[(float(entering.events+k),F.one_hot(token,num_classes=model.vocabulary)
                .to(model.embedding.weight.dtype)) for k,token in enumerate(tokens)])


def sequential(model,tokens,targets,entering,seed,force=None,record=None):
    count=[0];state=copy.deepcopy(entering);model.train()
    def race(scores,values=None):
        index=count[0];count[0]+=1
        value,delay,winner=(force_at_first_time(scores,values,force[1]) if force is not None and index==force[0]
                            else factorized_race(scores,values))
        if record is not None:record.append(dict(scores=scores,delay=delay.detach().clone(),winner=winner.detach().clone()))
        return value,delay,winner
    model.race=race
    try:
        with torch.random.fork_rng():
            torch.manual_seed(seed);logits,state=model.forward_chunk(tokens,state);end_rng=torch.get_rng_state().clone()
    finally:del model.race
    return F.cross_entropy(logits,targets,reduction='none'),logits,state,end_rng


def sequential_objective(model,tokens,targets,entering,seed):
    scores=[];losses,logits,state,end_rng=sequential(model,tokens,targets,entering,seed,record=scores)
    result=losses.sum();shadow=0
    for race,r in enumerate(scores):
        event=race//(model.depth*model.heads)
        with torch.no_grad():
            returns=torch.stack([sequential(model,tokens,targets,entering,seed,force=(race,i))[0][event:].sum()
                                 for i in range(model.pool)])
        result=result+(r['scores'].softmax(0)*returns).sum();shadow+=model.pool
    return result,logits,state,dict(shadow_lanes=shadow,shadow_events=shadow*len(tokens))


def batched_objective(model,tokens,targets,entering,seed):
    model.train();scores=[];observed=row(model,tokens,entering)
    logits,states,end_rng=batched_chunks(model,[observed],seed,[entering],record=scores)
    losses=F.cross_entropy(logits[0],targets,reduction='none');R=len(scores)
    forces=[(race,i) for race in range(R) for i in range(model.pool)]
    with torch.no_grad():
        alternative,_,_=batched_chunks(model,[observed]*len(forces),seed,[entering]*len(forces),forces)
        utilities=F.cross_entropy(alternative.reshape(-1,model.vocabulary),targets.repeat(len(forces)),
                                 reduction='none').reshape(len(forces),len(tokens))
        returns=torch.stack([utilities[j,race//(model.depth*model.heads):].sum()
                             for j,(race,i) in enumerate(forces)]).reshape(R,model.pool)
    result=losses.sum()+sum((s[0].softmax(0)*u).sum() for s,u in zip(scores,returns))
    return result,logits[0],states[0],dict(shadow_lanes=len(forces),shadow_events=len(forces)*len(tokens),
          factual_loss_sum=float(losses.detach().sum()))
