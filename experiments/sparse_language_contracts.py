"""Guarded numerical contracts for the integrated sparse temporal language model."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from sleeping_machines.sparse_race_language import SparseRaceLanguageModel,TemporalRoute

SOURCES=['sleeping_machines/sparse_race_language.py','experiments/sparse_language_contracts.py',
    'experiments/sparse_language_screen.py','sleeping_machines/race_language.py',
    'experiments/race_language_screen.py','experiments/race_language_contracts.py',
    'sleeping_machines/selective_stream_language.py','sleeping_machines/parallel_stream_language.py',
    'sleeping_machines/stream_language.py','sleeping_machines/event_state.py',
    'sleeping_machines/event_memory.py','sleeping_machines/operation_audit.py',
    'sleeping_machines/language_memory.py','experiments/parallel_event_language.py',
    'experiments/e120_shared_tasks.py']


def hashes():return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/parallel_language'/f'{a.tag}.json'
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique tag required')
    torch.set_num_threads(1);torch.manual_seed(6);started=time.perf_counter()
    model=SparseRaceLanguageModel(payload=8,depth=3,pool=2)
    tokens=torch.tensor([1,2,1,3,1,2,1,4,2,1,3,1,2,4,1,2])
    def run(parts,training=False,origin=0,sequence=tokens):
        torch.manual_seed(411);model.train(training);s=model.new_state();s.position=origin;rows=[]
        for l,r in parts:
            z,s=model.forward_chunk(sequence[l:r],s);rows.append(z)
        return torch.cat(rows),s
    whole,s=run([(0,len(tokens))])
    split,other=run([(0,5),(5,11),(11,len(tokens))])
    torch.testing.assert_close(whole,split,rtol=0,atol=0)
    shifted,_=run([(0,len(tokens))],origin=10000000)
    torch.testing.assert_close(whole,shifted,rtol=5e-5,atol=5e-6)
    teacher,trained=run([(0,len(tokens))],training=True)
    torch.testing.assert_close(whole,teacher,rtol=0,atol=0)
    altered=tokens.clone();altered[10:]=(altered[10:]+3)%27
    future,_=run([(0,len(tokens))],sequence=altered)
    torch.testing.assert_close(whole[:10],future[:10],rtol=0,atol=0)
    expected=len(tokens)*model.depth
    assert s.selected_updates==s.deliveries==expected
    assert s.candidate_scores==expected*model.pool
    assert s.counterfactual_values==0 and trained.counterfactual_values==expected*model.pool
    assert len(s.memories)==len(s.visited_units)<=expected<model.capacity_units
    assert all(t<=s.position for t in s.arrivals.values())
    loss=F.cross_entropy(teacher[:-1],tokens[1:]);loss.backward()
    summaries={}
    for name,parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(),name
            category=('query' if name.startswith('queries') else 'key' if '.key' in name else
                      'clock' if name.endswith('clock_bias') else 'memory' if '.raw_rate' in name else
                      'value' if '.input.' in name or '.output.' in name else 'other')
            summaries[category]=summaries.get(category,0.)+float(parameter.grad.norm())
    assert all(summaries.get(k,0)>0 for k in ('query','key','clock','memory','value')),summaries
    # Interior timing derivative is checked separately from the explicitly
    # surrogate counterfactual score teacher.
    torch.manual_seed(9);scores=torch.tensor([-.2,.4],requires_grad=True)
    values=torch.tensor([[1.,-.3],[.2,.8]],requires_grad=True)
    value,delay,winner=TemporalRoute.apply(scores,values)
    delay.backward(retain_graph=True)
    assert int((scores.grad!=0).sum())==1 and scores.grad[int(winner)]<0
    scores.grad=None;values.grad=None;value.sum().backward()
    assert abs(float(scores.grad.sum()))<1e-6
    result=dict(status='completed',args=vars(a),source_sha256=hashes(),
        partition_error=float((whole-split).detach().abs().max()),
        origin_error=float((whole-shifted).detach().abs().max()),
        training_forward_equals_winner_inference=True,causality=True,
        capacity_units=model.capacity_units,active_units_per_token=model.depth,
        candidate_scores=s.candidate_scores,counterfactual_values=trained.counterfactual_values,
        gradient_norm_totals=summaries,timing_gradient='passed',conserved_value_credit='passed',
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Integrated hard routing, sparse state updates, timed payloads and counterfactual learning; numerical contracts, not quality or energy evidence.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
