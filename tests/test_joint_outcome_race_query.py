import copy
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import joint_outcome_benchmark as J
import balanced_joint_benchmark as B
from balanced_joint_protocol import joint_examples
from sleeping_machines.native_stream_language import NativeStreamLanguageModel
from test_balanced_joint_learning import assert_tree_equal


def args(tag='test',credit='joint'):
    return J.parser().parse_args(['--tag',tag,'--payload','4','--depth','2','--fit-groups','1',
        '--dev-groups','1','--epochs','2','--update-targets','4','--read-credit',credit])


def test_zero_delivery_nests_parent_and_protects_generic_causal_outcomes():
    a=args();m=J.make_model(a);torch.manual_seed(a.seed);parent=NativeStreamLanguageModel(4,2,2,2)
    for n,t in parent.state_dict().items():torch.testing.assert_close(t,m.core.state_dict()[n],rtol=0,atol=0)
    row=joint_examples(groups=1)[1];m.eval();parent.eval()
    torch.manual_seed(81);x,s=m.forward_chunk(row['inputs'])
    torch.manual_seed(81);y,_=parent.forward_chunk(row['inputs'])
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    expected={}
    for prev,token in zip(row['inputs'],row['inputs'][1:]):expected[prev]=token
    assert s.outcomes==expected and s.outcome_writes==14
    assert s.outcomes[24]==0 and s.outcomes[25]==1
    changed=copy.deepcopy(row);changed['target']=0;changed['bits']=[9,9]
    z,_=B.episode(m,changed,19);original,_=B.episode(m,row,19)
    torch.testing.assert_close(z,original,rtol=0,atol=0)


def test_joint_route_derivative_is_exact_conditional_expected_risk():
    torch.manual_seed(31);m=J.make_model(args()).double()
    with torch.no_grad():m.interaction.normal_(std=.2);m.linear_values.normal_(std=.2)
    h=torch.randn(8,dtype=torch.float64);v=torch.randn(3,4,dtype=torch.float64);base=torch.tensor(.1,dtype=torch.float64)
    s1=torch.randn(3,dtype=torch.float64,requires_grad=True);s2=torch.randn(3,dtype=torch.float64,requires_grad=True)
    pair=m.pair_logits(h,base,v);losses=F.softplus(-pair);p1=s1.softmax(0);p2=s2.softmax(0)
    loss=(p1[:,None]*p2[None,:]*losses).sum()
    g1,g2=torch.autograd.grad(loss,(s1,s2))
    torch.testing.assert_close(g1,p1*((losses*p2[None,:]).sum(1)-loss),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(g2,p2*((losses*p1[:,None]).sum(0)-loss),rtol=1e-12,atol=1e-12)
    for i in range(3):
        hi=s1.detach().clone();lo=hi.clone();hi[i]+=1e-6;lo[i]-=1e-6
        numeric=(((hi.softmax(0)-lo.softmax(0))[:,None]*p2[None,:]*losses).sum())/2e-6
        torch.testing.assert_close(g1[i],numeric,rtol=1e-6,atol=1e-9)


@pytest.mark.parametrize('credit',['joint','local'])
def test_real_interrupted_driver_recovers_new_model_and_adam(tmp_path,credit):
    J.activate();a=args('continuous',credit);B.run(a,tmp_path)
    b=args('recovered',credit);b.stop_after_updates=1;B.run(b,tmp_path)
    b.resume=True;b.stop_after_updates=None;B.run(b,tmp_path)
    x=torch.load(tmp_path/'continuous.progress.pt',weights_only=False)
    y=torch.load(tmp_path/'recovered.progress.pt',weights_only=False)
    for k in ('online_model','optimizer','best_state','cursor','torch_rng','best'):assert_tree_equal(x[k],y[k])
    for k in ('final','activity','work','work_samples','selected_epoch'):assert_tree_equal(x['result'][k],y['result'][k])
    assert x['result']['work']['fitting_targets']==8 and x['result']['work']['optimizer_updates']==2


def test_trained_terminal_forward_matches_inference_and_chunking():
    a=args(credit='local');m=J.make_model(a);opt=torch.optim.Adam(m.parameters(),lr=.01)
    rows=joint_examples(groups=1);J.train_window(m,opt,rows,a,1)
    def replay(training,parts):
        m.train(training);torch.manual_seed(131);state=m.new_state();out=[];cursor=0
        with torch.no_grad():
            for n in parts:
                z,state=m.forward_chunk(rows[0]['inputs'][cursor:cursor+n],state);out.append(z);cursor+=n
        return torch.cat(out),state
    whole,s=replay(False,[15]);partition,t=replay(False,[4,4,7]);taught,_=replay(True,[15])
    torch.testing.assert_close(whole,partition,rtol=0,atol=0)
    torch.testing.assert_close(whole,taught,rtol=0,atol=0)
    assert s.outcomes==t.outcomes
