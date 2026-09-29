"""Content-memory nesting, exact credit, sparse-work and depth contracts."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e135_content_memory import ContentMemory, install
from e134_value_phase import build, SOURCE, hashes
from sleeping_machines.event_memory import linear_memory
from e117_serial_event_shd import batch, load_items


def reference(module, x, times, counts, keys, taus):
    """Quadratic audit oracle only; the candidate uses linear_memory."""
    pq,pk=module.features(x,module.query),module.features(x,module.key)
    kernel=pq@pk.T
    rows=[]
    for i in range(len(x)):
        eligible=(keys==keys[i]) & (torch.arange(len(x))<=i)
        weight=counts[:,None]*torch.exp(-(times[i]-times)[:,None]/taus[None,:])
        weight=weight*eligible[:,None]*kernel[i,:,None]
        rows.append(torch.einsum('ek,ed->kd',weight,x)/(weight.sum(0)[:,None]+1e-4))
    return torch.stack(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=Path('experiments/results/e135')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Invalid output')
    torch.set_num_threads(1);torch.manual_seed(135)
    x=torch.randn(17,8,dtype=torch.float64,requires_grad=True)
    times=torch.tensor([.01,.02,.02,.03,.05,.07,.09,.10,.10,.12,.15,.19,.23,.3,.4,.5,.6],dtype=torch.float64,requires_grad=True)
    counts=torch.linspace(.5,2.,17,dtype=torch.float64,requires_grad=True)
    keys=torch.arange(17)%3
    taus=torch.tensor([.03,.3,1.],dtype=torch.float64,requires_grad=True)
    module=ContentMemory(8).double()
    y,mass,work=module(x,times,counts,keys,taus)
    old,_,_=linear_memory(x,times,counts,keys,taus)
    nesting_error=float((y-old).detach().abs().max());assert nesting_error<3e-14
    teacher=torch.randn_like(y)
    qg,kg=torch.autograd.grad((y*teacher).sum(),(module.query,module.key))
    assert qg.norm()>1e-5 and kg.norm()<1e-12
    with torch.no_grad():module.query.add_(qg,alpha=-.03/float(qg.norm()))
    ys,_,_=module(x,times,counts,keys,taus,True)
    yp,_,_=module(x,times,counts,keys,taus)
    yr=reference(module,x,times,counts,keys,taus)
    reference_error=float((yp-yr).detach().abs().max());assert reference_error<3e-14
    assert float((ys-yp).detach().abs().max())<3e-14
    params=(x,times,counts,taus,module.query,module.key)
    actual=torch.autograd.grad((yp*teacher).sum(),params,retain_graph=True)
    expected=torch.autograd.grad((yr*teacher).sum(),params)
    credit_errors=[float((p-q).abs().max()) for p,q in zip(actual,expected)]
    assert max(credit_errors)<2e-12 and actual[-1].norm()>1e-6
    calls=module.pop_calls()
    assert calls[0]['scan_scalar_multiply_adds']==work*3*(5*9)
    # The augmented width is charged even though scan composition count is identical.
    assert calls[0]['scan_scalar_multiply_adds']>calls[0]['plain_scan_scalar_multiply_adds']
    plain,_,_=build('fixed');plain.eval()
    model=copy.deepcopy(plain);install(model);model.eval()
    rows=load_items(40,.01,4,'fit_spk',6);data=batch(rows)
    z0,_,_,tr0=plain(*data[:4],len(rows),trace=True)
    z,_,_,tr=model(*data[:4],len(rows),trace=True)
    logits_error=float((z-z0).detach().abs().max());assert logits_error<3e-5
    assert all(torch.equal(p['winner'],q['winner']) and torch.equal(p['times'],q['times']) for p,q in zip(tr0,tr))
    for layer in model.layers:layer.memory.pop_calls()
    payload_bound=float(model.embedding.weight.detach().abs().max())+1.
    ly=model.layers[0].memory.input_lipschitz_bound(payload_bound)
    a_l=[ly*(2 if j in model.global_context_layers else 1) for j in range(8)]
    assert all(model.layers[j].alpha*a_l[j]<1 for j in range(8))
    with torch.no_grad():
        for layer in model.layers:layer.memory.query.add_(torch.randn_like(layer.memory.query)*.1)
        zp,_,_,trp=model(*data[:4],len(rows),trace=True)
    assert all(torch.equal(p['winner'],q['winner']) and torch.equal(p['times'],q['times']) for p,q in zip(tr0,trp))
    result={'status':'completed','mean_nesting_max_error':nesting_error,
      'linear_vs_direct_max_error':reference_error,'credit_errors_payload_time_count_tau_query_key':credit_errors,
      'balanced_query_gradient_norm':float(qg.norm()),'balanced_key_gradient_norm':float(kg.norm()),
      'key_gradient_after_query_step':float(actual[-1].norm()),'scan_work_ledger_example':calls[0],
      'parent_logits_nesting_max_error':logits_error,'actual_keys_winners_clocks_preserved':True,
      'payload_bound_at_initial_embedding':payload_bound,'content_input_lipschitz_bound':ly,
      'conditional_value_state_transport_bounds':[math.prod(1-model.layers[j].alpha*a_l[j] for j in range(8)),
                                                    math.prod(1+model.layers[j].alpha*a_l[j] for j in range(8))],
      'parent_checkpoint_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
      'source_sha256':{**hashes(__file__), 'experiments/e135_content_memory.py':hashlib.sha256(Path('experiments/e135_content_memory.py').read_bytes()).hexdigest()},
      'scope':'Contracts and a candidate layer, not speech accuracy or energy. Bounds require fixed schedules and the stated payload/projection bounds.'}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
