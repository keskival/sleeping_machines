"""One actual alternative replay for native pools, retaining original timing."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import dvs_state_choice_credit_benchmark as B
from sleeping_machines.paired_route_credit import alternative_probabilities,paired_choice_credit,sample_alternative

N,S,K=B.N,B.S,B.K
BASE_PARSER=N.parser


def parser():
    p=BASE_PARSER();p.add_argument('--alternative-exploration',type=float,default=.1);return p


def sources():
    names=['experiments/dvs_paired_choice_benchmark.py','experiments/dvs_paired_choice_contracts.py',
        'experiments/dvs_state_choice_credit_benchmark.py','experiments/dvs_state_clock_credit_benchmark.py',
        'experiments/dvs_counterfactual_route_audit.py','sleeping_machines/paired_route_credit.py',
        'sleeping_machines/batched_addressed_fit.py','experiments/theory/86_paired_memory_route_credit.md']
    return {**S.BASE_SOURCES(),**{name:N.sha(ROOT/name) for name in names}}


def corrected_loss(model,rows,seed,epoch,exploration=.1,audit=False,alternative=None):
    if model.pool<2:raise ValueError('An eligible alternative is required')
    head=(epoch-1)%model.heads;depth=((epoch-1)//model.heads)%model.depth;site=9*model.depth+depth
    targets=torch.tensor([r['target'] for r in rows],device=model.embedding.weight.device)
    with B.trace_site(site,head,audit) as records:logits,state,_=K.forward(model,rows,seed)
    box=records[0];factual=F.cross_entropy(logits,targets,reduction='none')
    with torch.no_grad():
        pi=box['scores'][:,head].double().softmax(-1)
        proposal=alternative_probabilities(pi,box['winner'],exploration)
        selected=sample_alternative(proposal) if alternative is None else alternative
        with S.trace_site(site,head,selected) as shadows:z,shadow_state,_=K.forward(model,rows,seed)
        counterfactual=F.cross_entropy(z,targets,reduction='none')
        box['credit']=paired_choice_credit(pi,box['winner'],selected,proposal,factual.detach(),counterfactual)
    return factual.sum(),state,dict(record=box,logits=logits,alternative=selected,proposal=proposal,
        alternative_loss=counterfactual,shadow=shadows[0],shadow_state=shadow_state,site=site,head=head)


def train_window(model,optimizer,rows,a,epoch,trace=False):
    previous=S.corrected_loss
    S.corrected_loss=lambda m,r,s,e:corrected_loss(m,r,s,e,a.alternative_exploration)
    try:return S.train_window(model,optimizer,rows,a,epoch,trace)
    finally:S.corrected_loss=previous


@contextmanager
def activate():
    previous=N.parser,N.sources,N.train_window
    N.parser,N.sources,N.train_window=parser,sources,train_window
    try:yield
    finally:N.parser,N.sources,N.train_window=previous


def run(a,directory=None):
    with activate():result=N.run(a,directory)
    if result['status']=='completed':
        result['credit_protocol']=dict(kind='paired_state_choice',event=9,corrected_heads_per_clip_per_window=1,
            gradient='Paired actual-loss difference with recorded proposal propensity; native timing retained',
            full_shadow_forwards_per_window=1,alternative_exploration=a.alternative_exploration,
            auxiliary_proposal_exponential_draws_per_fit_target=a.pool,
            importance_bound=1/(1-a.alternative_exploration),timing_gradient_retained=True,
            inference_architecture_unchanged=True,full_model_gradient_exact=False,
            activity_scope='Factual plus full alternative replay keys, candidates and state commits; proposal arithmetic paid, RNG work separate')
        folder=Path(directory) if directory else ROOT/'experiments/results/dvs_native'
        (folder/(a.tag+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':run(parser().parse_args())
