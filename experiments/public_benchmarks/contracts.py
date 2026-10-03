"""Guarded content-shape, variable-length, gradients and warm-Adam port contracts."""
import argparse,copy,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch
from torch.nn import functional as F
from experiments.public_benchmarks.run import encode,source_hashes,atomic
from experiments.public_benchmarks.data import load_train,contracts
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.batched_episodes import batched_logits
from sleeping_machines.compiled_episodes import compiled_logits

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.set_num_threads(1);started=time.perf_counter();checks=[contracts()]
    for name in ['JapaneseVowels','ECG200','PenDigits']:
        raw,mapping,_=load_train(name);n=len(raw[0]['values'][0]);horizon=max(len(r['values']) for r in raw)
        rows=encode(raw[:2],np.zeros(n,np.float32),np.ones(n,np.float32),horizon)
        for r in rows:r['events']=r['events'][:3]+r['events'][-1:]
        # Full episode terminal-query interface with public task dimensions.
        torch.manual_seed(613);model=fast_class(AddressedEventHeads)(sources=1,content_dim=n+1,classes=len(mapping),payload=4,depth=2,heads=2,pool=2).double()
        outputs=[]
        for function in [batched_logits,compiled_logits]:
            m=copy.deepcopy(model);opt=torch.optim.Adam(m.parameters(),lr=.003)
            histories=[]
            for _ in range(2):
                opt.zero_grad();z=function(m,rows,123,route_credit='linear')
                loss=F.cross_entropy(z,torch.tensor([r['target'] for r in rows]));loss.backward()
                grad=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()]
                histories.append((z.detach().clone(),grad));torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            outputs.append((histories,[p.detach().clone() for p in m.parameters()],opt.state_dict()))
        left,right=outputs
        for (za,ga),(zb,gb) in zip(left[0],right[0]):
            assert torch.allclose(za,zb,rtol=1e-8,atol=1e-9)
            assert all(torch.allclose(a,b,rtol=1e-8,atol=1e-9) for a,b in zip(ga,gb))
        assert all(torch.allclose(a,b,rtol=1e-8,atol=1e-9) for a,b in zip(left[1],right[1]))
        checks.append(name+': eager/compiled terminal logits, every gradient and two actual Adam updates agree')
    out=ROOT/'experiments/results/public_benchmarks'/(args.tag+'.json')
    out.parent.mkdir(exist_ok=True);assert not out.exists()
    atomic(out,dict(status='completed',contracts=checks,source_sha256=source_hashes(),wall_s=time.perf_counter()-started,
                   scope='Numerical port admission only; no public test scores or benchmark advantage.'))
    print(json.dumps(dict(status='completed',contracts=checks)),flush=True)
