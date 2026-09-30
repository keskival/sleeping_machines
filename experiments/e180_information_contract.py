"""Exact finite checks of stopped-token Fisher and noisy-update risk identities."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sleeping_machines.prefix_tokenizer import CompletePrefixTokenizer


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    out=Path('experiments/results/e180')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique output required')
    torch.set_num_threads(1);start=time.perf_counter()
    codec=CompletePrefixTokenizer([(0,0),(0,1),(1,)],alphabet=2)
    q=torch.tensor([.23,.31,.46],dtype=torch.float64)
    def posterior(prefix):
        mask=q.new_zeros(3);mask[codec.descendants[prefix]]=1
        return q*mask/(q*mask).sum()
    pieces=q.new_zeros((3,3));leaf=q.new_zeros((3,3));cross=q.new_zeros((3,3))
    for j,phrase in enumerate(codec.phrases):
        increments=[posterior(phrase[:k+1])-posterior(phrase[:k]) for k in range(len(phrase))]
        for score in increments:pieces+=q[j]*torch.outer(score,score)
        for i in range(len(increments)):
            for k in range(i+1,len(increments)):cross+=q[j]*torch.outer(increments[i],increments[k])
        total=torch.stack(increments).sum(0);leaf+=q[j]*torch.outer(total,total)
    fisher=torch.diag(q)-torch.outer(q,q)
    mu=torch.tensor([.2,-.3],dtype=torch.float64)
    H=torch.tensor([[1.4,.2],[.2,.9]],dtype=torch.float64)
    P=torch.diag(torch.tensor([.7,1.2],dtype=torch.float64));eta=.15
    noises=[torch.tensor([.4*x+.1*y,.2*x-.3*y],dtype=torch.float64)
            for x,y in itertools.product((-1.,1.),repeat=2)]
    Sigma=torch.stack([torch.outer(v,v) for v in noises]).mean(0)
    risks=[];signals=[]
    for first,second in itertools.product(noises,repeat=2):
        g1,g2=mu+first,mu+second;g=(g1+g2)/2;d=-eta*(P@g)
        risks.append(mu@d+.5*d@H@d)
        signals.append(g1@P@g2)
    predicted=-eta*(mu@P@mu)+eta**2/2*(mu@P@H@P@mu+torch.trace(H@P@(Sigma/2)@P))
    result=dict(status='completed',fisher_increment_error=float((pieces-fisher).abs().max()),
        fisher_leaf_error=float((leaf-fisher).abs().max()),orthogonal_increment_error=float(cross.abs().max()),
        expected_quadratic_risk_error=abs(float(torch.stack(risks).mean()-predicted)),
        expected_cross_sample_signal_error=abs(float(torch.stack(signals).mean()-mu@P@mu)),
        protocol='Exact enumeration, float64; finite prefix dictionary and independent four-point gradient noise. No language/speech quality claim.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),wall_s=time.perf_counter()-start)
    assert max(result[k] for k in result if k.endswith('_error'))<1e-12
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
