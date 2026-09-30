"""Completed-checkpoint gradient geometry and restored finite virtual steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e137_compact_scattering_model import CompactScatteringClassifier
from e117_serial_event_shd import batch,load_items


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    base=Path('experiments/results/e137');out=base/(a.tag+'.json')
    source=base/'compact_learned_d12_r16_n1024_s6_e3_resume_20260930.json'
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    training=json.loads(source.read_text());assert training['status']=='completed'
    torch.set_num_threads(1);model=CompactScatteringClassifier()
    checkpoint=source.with_suffix('.pt');saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
    model.load_state_dict(saved['state_dict']);model.eval()
    named={n:p for n,p in model.named_parameters() if p.requires_grad}
    groups={'angles':[n for n in named if n.startswith('angle_')],
            'embedding':[n for n in named if n.startswith('embedding.')],
            'query':[n for n in named if n.startswith('head.')]}
    items={'fit':load_items(40,.01,64,'fit_spk',6),'held':load_items(40,.01,64,'val_spk',7)}
    assert [r[4] for r in items['fit']]==training['fit_ids'][:64]
    assert [r[4] for r in items['held']]==training['held_ids'][:64]
    gradients={split:{group:[] for group in groups} for split in items};losses={}
    for split,rows in items.items():
        loss_sum=0.
        for pos in range(0,len(rows),4):
            data=batch(rows[pos:pos+4]);logits,_,_,_=model(*data[:4],4)
            loss=F.cross_entropy(logits,data[-1]);loss_sum+=float(loss.detach())*4
            values=torch.autograd.grad(loss,list(named.values()))
            by_name=dict(zip(named,values))
            for group,names in groups.items():
                gradients[split][group].append(torch.cat([by_name[n].detach().flatten() for n in names]).double())
        losses[split]=loss_sum/len(rows)
        for group in groups:gradients[split][group]=torch.stack(gradients[split][group])
    statistics={}
    for group in groups:
        statistics[group]={}
        for split in items:
            samples=gradients[split][group];n=len(samples);mean=samples.mean(0)
            second=samples.square().sum(1).mean();mean_square=mean.square().sum()
            pair=(n*mean_square-second)/(n-1)
            statistics[group][split]={'batches':n,'parameters':samples.shape[1],
              'mean_batch_squared_norm':float(second),'squared_mean_gradient':float(mean_square),
              'distinct_batch_inner_product_mean':float(pair),
              'mean_direction_fraction':float(mean_square/second),
              'scope':'Descriptive sampled minibatches, Euclidean parameter metric. Distinct-batch statistic estimates transferable first-order signal under an IID assumption; it can be negative in a finite sample.'}
        left=gradients['fit'][group];right=gradients['held'][group]
        products=left@right.T;lf=left.mean(0);rh=right.mean(0)
        statistics[group]['cross_speaker']={'mean_gradient_inner_product':float(lf@rh),
          'mean_gradient_cosine':float((lf@rh)/(lf.norm()*rh.norm()).clamp_min(1e-30)),
          'negative_batch_pair_fraction':float((products<0).double().mean()),
          'batch_pair_inner_product_quantiles':torch.quantile(products.flatten(),torch.tensor([0.,.1,.5,.9,1.],dtype=torch.float64)).tolist()}
    @torch.no_grad()
    def nll(rows):
        total=0.
        for pos in range(0,len(rows),4):
            data=batch(rows[pos:pos+4]);z,_,_,_=model(*data[:4],4)
            total+=float(F.cross_entropy(z,data[-1],reduction='sum'))
        return total/len(rows)
    probes={}
    for group in ('angles','query'):
        names=groups[group];mean=gradients['fit'][group].mean(0)
        norm=mean.norm();step=.001;direction=mean/norm
        old={n:named[n].detach().clone() for n in names};offset=0
        with torch.no_grad():
            for name in names:
                p=named[name];size=p.numel();p.add_((-step*direction[offset:offset+size]).reshape_as(p).to(p));offset+=size
        probes[group]={'parameter_step_l2':step,
          'fit_predicted_nll_change':float(-step*norm),
          'held_predicted_nll_change':float(-step*(gradients['held'][group].mean(0)@direction)),
          'fit_actual_nll_change':nll(items['fit'])-losses['fit'],
          'held_actual_nll_change':nll(items['held'])-losses['held']}
        with torch.no_grad():
            for name in names:named[name].copy_(old[name])
        assert all(torch.equal(named[name],old[name]) for name in names)
    result={'status':'completed','fit_held_nll':losses,'gradient_statistics':statistics,'finite_virtual_steps':probes,
       'sample_ids':{split:[r[4] for r in rows] for split,rows in items.items()},
       'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),source,checkpoint]},
       'scope':'Frozen final checkpoint, first 64 declared fit/held utterances, 16 minibatches each. Virtual perturbations are restored and do not train on development labels. Not population/seed uncertainty or an optimizer convergence proof.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
