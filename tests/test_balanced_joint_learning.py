import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import balanced_joint_benchmark as B
from balanced_joint_protocol import joint_examples,validate_pairs


def arguments(tag='test',kind='native',payload=2,depth=2):
    return B.parser().parse_args(['--tag',tag,'--model',kind,'--payload',str(payload),
        '--depth',str(depth),'--fit-groups','2','--dev-groups','2','--epochs','2'])


def assert_tree_equal(a,b):
    assert type(a)==type(b)
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else:assert a==b


def test_root_and_all_query_counts_match_with_actual_prefix_updates():
    fit=joint_examples(groups=2,seed=83);dev=joint_examples(groups=2,seed=97)
    assert validate_pairs(dev)['identical_symbol_totals']
    controls=B.count_controls(fit,dev)
    assert controls['all_query_count_vectors_including_root_equal']
    assert len(controls['rows'])==36
    for row in controls['rows']:assert row['query_bits']==pytest.approx(1.,abs=1e-12)
    damaged=copy.deepcopy(dev);damaged[0]['inputs'][2]=0
    with pytest.raises(ValueError):validate_pairs(damaged)


def test_full_shape_taps_nest_native_forward_and_parent_gradients():
    a=arguments(payload=8,depth=4);native=B.make_model(a)
    a.model='tapped';tapped=B.make_model(a);native.train();tapped.train()
    row=joint_examples(groups=1)[1]
    x,s=B.episode(native,row,611);y,t=B.episode(tapped,row,611)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    F.cross_entropy(x[None],torch.tensor([row['target']])).backward()
    F.cross_entropy(y[None],torch.tensor([row['target']])).backward()
    other=dict(tapped.named_parameters())
    for n,p in native.named_parameters():
        if p.grad is None:assert other[n].grad is None
        else:torch.testing.assert_close(p.grad,other[n].grad,rtol=0,atol=0)
    assert s.events==t.events==15 and t.storage()['tap_buffer_vectors']>0
    assert any(float(m.tap.weight.grad.norm())>0 for m in tapped.channel_mix)
    changed=copy.deepcopy(row);changed['target']=1-row['target'];changed['bits']=[9,9]
    z,_=B.episode(tapped,changed,611)
    torch.testing.assert_close(z,y,rtol=0,atol=0)


@pytest.mark.parametrize('kind',['native','tapped'])
def test_real_driver_recovers_optimizer_selection_and_cursor(tmp_path,kind):
    a=arguments('whole',kind);B.run(a,tmp_path)
    b=arguments('recovered',kind);b.stop_after_updates=1
    B.run(b,tmp_path);b.resume=True;b.stop_after_updates=None;B.run(b,tmp_path)
    x=torch.load(tmp_path/'whole.progress.pt',weights_only=False)
    y=torch.load(tmp_path/'recovered.progress.pt',weights_only=False)
    for k in ('online_model','optimizer','best_state','initial','cursor','torch_rng','best'):
        assert_tree_equal(x[k],y[k])
    for k in ('activity','work_samples','work','final','selected_epoch','parameter_delta'):
        assert_tree_equal(x['result'][k],y['result'][k])
    assert x['result']['work']['fitting_targets']==16
    assert x['result']['work']['optimizer_updates']==2


def test_normalization_matches_mean_loss_adam_on_full_episode_targets():
    a=arguments();m=B.make_model(a);other=copy.deepcopy(m)
    rows=joint_examples(groups=1);opt=torch.optim.Adam(m.parameters(),lr=a.lr)
    ropt=torch.optim.Adam(other.parameters(),lr=a.lr)
    B.train_window(m,opt,rows,a,1)
    other.train();losses=[]
    for row in rows:
        z,_=B.episode(other,row,100000+a.seed+10000)
        losses.append(F.cross_entropy(z[None],torch.tensor([row['target']])))
    torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(other.parameters(),1.,error_if_nonfinite=True);ropt.step()
    for p,q in zip(m.parameters(),other.parameters()):torch.testing.assert_close(p,q,rtol=1e-6,atol=1e-7)


def test_trained_tap_predictions_are_causal_and_chunk_invariant():
    a=arguments(kind='tapped');m=B.make_model(a);opt=torch.optim.Adam(m.parameters(),lr=a.lr)
    row=joint_examples(groups=1)[0];B.train_window(m,opt,[row],a,1);m.eval()
    def replay(seq,parts):
        torch.manual_seed(71);state=m.new_state();outputs=[];cursor=0
        with torch.no_grad():
            for n in parts:
                z,state=m.forward_chunk(seq[cursor:cursor+n],state);outputs.append(z);cursor+=n;state.detach()
        return torch.cat(outputs)
    seq=torch.tensor(row['inputs']);whole=replay(seq,[15]);split=replay(seq,[4,4,7])
    torch.testing.assert_close(whole,split,rtol=0,atol=0)
    changed=seq.clone();changed[10:]=(changed[10:]+1)%27
    torch.testing.assert_close(whole[:10],replay(changed,[15])[:10],rtol=0,atol=0)


def test_full_shape_actual_adam_moments_and_next_update_recover():
    a=arguments(kind='tapped',payload=8,depth=4);m=B.make_model(a)
    opt=torch.optim.Adam(m.parameters(),lr=a.lr);rows=joint_examples(groups=1)
    B.train_window(m,opt,rows,a,1)
    restored=copy.deepcopy(m);other=torch.optim.Adam(restored.parameters(),lr=a.lr)
    other.load_state_dict(copy.deepcopy(opt.state_dict()))
    B.train_window(m,opt,rows,a,2);B.train_window(restored,other,rows,a,2)
    assert_tree_equal(m.state_dict(),restored.state_dict())
    assert_tree_equal(opt.state_dict(),other.state_dict())
    x=B.evaluate(m,rows);y=B.evaluate(restored,rows);assert_tree_equal(x,y)
