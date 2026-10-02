"""Full-shape causal/nesting and next-Adam recovery contracts for memory repairs."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from parallel_head_gradient_accumulation import GradientAccumulator
from sleeping_machines.native_stream_language import NativeStreamLanguageModel
from sleeping_machines.context_addressed_memory import ContextAddressedNativeModel
from sleeping_machines.dilated_delay_taps import TappedNativeStreamLanguageModel


def contract(kind):
    torch.manual_seed(117)
    parent=NativeStreamLanguageModel(payload=16,depth=8,pool=2,heads=2)
    cls=ContextAddressedNativeModel if kind=='addressed' else TappedNativeStreamLanguageModel
    model=cls(payload=16,depth=8,pool=2,heads=2)
    model.load_state_dict({**model.state_dict(),**parent.state_dict()})
    tokens=torch.tensor(([1,2,1,3,1,2,4,1,2,1,5,1,7,1,2,3]*9)[:128])
    rng=torch.get_rng_state(); parent.train(); model.train()
    z,_=parent.forward_chunk(tokens[:16]);F.cross_entropy(z,tokens[1:17]).backward()
    torch.set_rng_state(rng);q,_=model.forward_chunk(tokens[:16]);F.cross_entropy(q,tokens[1:17]).backward()
    torch.testing.assert_close(z,q,rtol=0,atol=0)
    for name,p in parent.named_parameters():
        grad=dict(model.named_parameters())[name].grad
        if p.grad is None:assert grad is None or float(grad.abs().sum())==0
        else:torch.testing.assert_close(p.grad,grad,rtol=0,atol=0)
    model.zero_grad(set_to_none=True)
    opt=torch.optim.Adam(model.parameters(),lr=.002);learner=GradientAccumulator(model,opt,.002)
    state=model.new_state()
    for begin,end in ((0,16),(16,32),(32,48),(48,64)):
        _,state,_=learner.accumulate(tokens[begin:end],tokens[begin+1:end+1],state)
        assert state.packed_storage()['differentiable_entries']==0
        assert all(not x.requires_grad and not t.requires_grad for x,t in state.contexts.values())
    raw={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    learner.normalize()
    for n,p in model.named_parameters():
        if p.grad is not None:torch.testing.assert_close(p.grad,raw[n]/64,rtol=0,atol=0)
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);learner.step()
    assert learner.total_targets==64 and learner.updates==1 and learner.pending_targets==0
    # Serialization preserves *all* temporal state, the newly added memories,
    # actual optimizer moments and RNG at a completed optimizer boundary.
    buffer=io.BytesIO()
    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,
        rng=torch.get_rng_state(),total=learner.total_targets,updates=learner.updates),buffer)
    buffer.seek(0);saved=torch.load(buffer,weights_only=False)
    restored=copy.deepcopy(model);restored.load_state_dict(saved['model'])
    ropt=torch.optim.Adam(restored.parameters(),lr=.002);ropt.load_state_dict(saved['optimizer'])
    recovered=GradientAccumulator(restored,ropt,.002)
    recovered.total_targets=saved['total'];recovered.updates=saved['updates']
    other=saved['state'];predictions=[];replayed=[]
    torch.set_rng_state(saved['rng'])
    for begin,end in ((64,80),(80,96),(96,112),(112,127)):
        _,state,p=learner.accumulate(tokens[begin:end],tokens[begin+1:end+1],state);predictions.append(p)
    if kind=='addressed':
        learned=dict(read_gradient_norm=float(model.memory_read_map.weight.grad.norm()),
            write_gradient_norm=float(model.memory_write_map.weight.grad.norm()))
        assert all(v>0 for v in learned.values())
    else:
        learned=dict(tap_gradient_norms=[float(m.tap.weight.grad.norm()) for m in model.channel_mix],
            delay_gradient_norms=[float(m.raw_delay.grad.abs()) for m in model.channel_mix])
        assert all(v>0 for v in learned['tap_gradient_norms'])
        assert any(v>0 for v in learned['delay_gradient_norms'])
    learner.update()
    torch.set_rng_state(saved['rng'])
    for begin,end in ((64,80),(80,96),(96,112),(112,127)):
        _,other,p=recovered.accumulate(tokens[begin:end],tokens[begin+1:end+1],other);replayed.append(p)
    recovered.update()
    torch.testing.assert_close(torch.cat(predictions),torch.cat(replayed),rtol=0,atol=0)
    for p,q in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    for p,q in zip(opt.state.values(),ropt.state.values()):
        assert p.keys()==q.keys()
        for name in p:torch.testing.assert_close(p[name],q[name],rtol=0,atol=0)
    assert learner.total_targets==recovered.total_targets==127
    assert learner.updates==recovered.updates==2
    # Nonzero trained repair must remain causal and independent of chunking.
    def forward(sequence,chunks):
        s=model.new_state();out=[];begin=0;torch.manual_seed(431);model.eval()
        with torch.no_grad():
            for size in chunks:
                q,s=model.forward_chunk(sequence[begin:begin+size],s);out.append(q);begin+=size;s.detach()
        return torch.cat(out)
    whole=forward(tokens[:64],[64]);partition=forward(tokens[:64],[16]*4)
    torch.testing.assert_close(whole,partition,rtol=0,atol=0)
    changed=tokens[:64].clone();changed[48:]=(changed[48:]+1)%27
    future=forward(changed,[64]);torch.testing.assert_close(whole[:48],future[:48],rtol=0,atol=0)
    return dict(model=kind,payload=16,depth=8,heads=2,pool=2,
        zero_repair_forward_and_parent_gradients_bitwise=True,
        actual_target_weighted_gradient_normalization=True,
        bitwise_next_adam_parameters_predictions_and_moments=True,
        trained_repair_causal_and_chunk_invariant=True,learned_repair=learned,
        final_state_storage=state.storage(),all_added_state_detached=True,
        diagnostic_targets_per_replica=127,optimizer_updates_per_replica=2,
        credit_events=16,optimizer_window=64,partial_window=63,
        scope='Numerical prerequisite, not useful-feature or quality evidence; no long-fit driver recovery claimed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unique unused plain tag required')
    started=time.perf_counter();torch.set_num_threads(1);rows=[]
    for kind in ('addressed','tapped'):
        rows.append(contract(kind));print(json.dumps(dict(completed=kind)),flush=True)
    names=['experiments/deep_memory_optimizer_contracts.py','experiments/parallel_head_gradient_accumulation.py',
           'sleeping_machines/context_addressed_memory.py','sleeping_machines/dilated_delay_taps.py',
           'sleeping_machines/native_stream_language.py','sleeping_machines/addressed_event_heads.py',
           'sleeping_machines/parallel_head_race_language.py','sleeping_machines/sparse_race_language.py',
           'sleeping_machines/parallel_stream_language.py','sleeping_machines/packed_episodic_race_language.py',
           'sleeping_machines/indexed_episodic_race_language.py']
    result=dict(status='completed',contracts=rows,official_test_read=False,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(completed=a.tag,wall_s=result['wall_s'],max_rss_kb=result['max_rss_kb'])),flush=True)


if __name__=='__main__':main()
