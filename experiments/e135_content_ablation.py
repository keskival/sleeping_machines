"""Remove learned content retrieval from a completed checkpoint, preserving values."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e134_value_phase import build
from e135_content_memory import install
from e117_serial_event_shd import batch, load_items
from e118_race_carrier_shd import evaluate
from sleeping_machines.event_memory import linear_memory


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e135')/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    path=Path('experiments/results/e135/content_full_value_s6_20260929.json')
    r=json.loads(path.read_text());assert r['status']=='completed'
    torch.set_num_threads(1)
    model,_,_=build('fixed');install(model)
    saved=torch.load(path.with_suffix('.pt'),weights_only=False,map_location='cpu')
    model.load_state_dict(saved['state_dict']);model.eval()
    payload_bound=float(model.embedding.weight.detach().abs().max())+1.
    ly=model.layers[0].memory.input_lipschitz_bound(payload_bound)
    row_gains=[]
    for layer in model.layers:
        w=layer.value.detach()/layer.value.detach().abs().sum(-1,keepdim=True).clamp_min(1)
        gain=w[:,:,:layer.dim].abs().sum(-1)+ly*w[:,:,layer.dim:2*layer.dim].abs().sum(-1)
        if layer.bridge_value is not None:
            bw=layer.bridge_value.detach()/layer.bridge_value.detach().abs().sum(-1,keepdim=True).clamp_min(1)
            gain=gain+ly*bw[:,:,:layer.dim].abs().sum(-1)
        row_gains.append(float(gain.max()))
    assert all(layer.alpha*row_gains[j]<1 for j,layer in enumerate(model.layers))
    plain=copy.deepcopy(model)
    for layer in plain.layers:
        del layer._modules['memory']
        layer.memory=linear_memory
    held=load_items(40,.01,512,'val_spk',7)
    for n,rows in (('dev_original',held[:256]),('dev_additional',held[256:])):
        assert r['ids'][n]==[item[4] for item in rows]
    original={n:r['final'][n] for n in ('dev_original','dev_additional')}
    neutral={n:evaluate(plain,rows,4) for n,rows in (('dev_original',held[:256]),('dev_additional',held[256:]))}
    for n in neutral:
        assert neutral[n]['winner_counts']==original[n]['winner_counts']
        assert neutral[n]['mean_added_delay_ms']==original[n]['mean_added_delay_ms']
    fit=load_items(40,.01,64,'fit_spk',6)
    max_logit_change=0.;mean_abs_change=0.;query_grads=torch.zeros(8);key_grads=torch.zeros(8)
    for start in range(0,len(fit),4):
        rows=fit[start:start+4];data=batch(rows)
        logits,_,_,_=model(*data[:4],len(rows))
        with torch.no_grad():base=plain(*data[:4],len(rows))[0]
        max_logit_change=max(max_logit_change,float((logits-base).detach().abs().max()))
        mean_abs_change+=float((logits-base).detach().abs().mean())*len(rows)
        model.zero_grad(set_to_none=True);F.cross_entropy(logits,data[-1]).backward()
        for j,layer in enumerate(model.layers):
            query_grads[j]+=layer.memory.query.grad.norm().detach()/16
            key_grads[j]+=layer.memory.key.grad.norm().detach()/16
            layer.memory.pop_calls()
    def row(parts):
        correct=sum(p['correct'] for p in parts.values());n=sum(p['n'] for p in parts.values())
        return {'held_correct':correct,'held_n':n,'held_accuracy':correct/n,
          'held_nll':sum(p['nll']*p['n'] for p in parts.values())/n}
    result={'status':'completed','learned_content':row(original),'content_removed_same_values':row(neutral),
      'maximum_clean_fit_logit_change_64':max_logit_change,'mean_absolute_clean_fit_logit_change_64':mean_abs_change/len(fit),
      'clean_fit_content_query_gradient_norms':query_grads.tolist(),'clean_fit_content_key_gradient_norms':key_grads.tolist(),
      'partitioned_row_gains':row_gains,
      'partitioned_conditional_transport_bounds':[math.prod(1-layer.alpha*row_gains[j] for j,layer in enumerate(model.layers)),
                                                   math.prod(1+layer.alpha*row_gains[j] for j,layer in enumerate(model.layers))],
      'partitioned_bound_scope':'Global max-norm certificate within the payload/projection bounds and fixed schedules; uses actual normalized direct/memory/global row partitions. Not a parameter-gradient or optimization bound.',
      'original_held_quality_reused':True,'actual_winners_clocks_unchanged':True,
      'held_content_removed':neutral,'checkpoint_sha256':hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest(),
      'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),path,Path('experiments/e135_content_memory.py'),Path('sleeping_machines/shared_event.py'),Path('sleeping_machines/event_memory.py')]},
      'scope':'Frozen-checkpoint intervention on held speakers. Values/head/keys unchanged; content kernel removed. Fitting-only gradients on 64 clean utterances, no update. Not a retrained plain-memory control or an official-test result.'}
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('held_content_removed','source_sha256')}),flush=True)


if __name__=='__main__':main()
