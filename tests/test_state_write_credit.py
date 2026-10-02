"""Read-only forward/gradient contracts; actual optimizer recovery is queued."""
import copy

import torch
from torch.nn import functional as F

from experiments.native_event_tasks import episodes
from experiments.state_credit_contracts import predict
from sleeping_machines.split_event_heads import SplitEventHeads
from sleeping_machines.state_write_credit import StateCreditEventHeads,StateWriteRoute


def test_same_message_and_content_still_credit_the_write_address():
    scores=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    values=torch.zeros(2,1,dtype=torch.float64,requires_grad=True)
    old=torch.zeros(2,1,dtype=torch.float64,requires_grad=True)
    new=torch.ones(2,1,dtype=torch.float64,requires_grad=True)
    times=torch.zeros(2,dtype=torch.float64,requires_grad=True)
    arrival=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
    with torch.random.fork_rng():
        torch.manual_seed(17);value,delay,winner,memory,_=StateWriteRoute.apply(scores,values,old,new,times,arrival,1.)
        memory[1].sum().backward()
    assert scores.grad[0]<0 and scores.grad[1]>0
    torch.testing.assert_close(scores.grad.sum(),torch.tensor(0.,dtype=torch.float64),atol=1e-14,rtol=0)
    assert old.grad[int(winner)]==0
    assert new.grad.sum()==int(winner)


def test_linear_state_credit_expectation_is_the_exact_categorical_gradient():
    # Integrate T analytically and enumerate W; no noisy convergence test.
    rates=torch.tensor([.3,1.4,2.]);p=rates/rates.sum()
    old=torch.tensor([[.4,-.2],[.1,.8],[-.5,.3]])
    new=torch.tensor([[.6,.1],[-.4,.7],[.8,.2]])
    adjoint=torch.tensor([[.2,.9],[-.4,.3],[.7,-.1]])
    utility=(adjoint*(new-old)).sum(1);c=p*(utility-utility.mean())
    estimated=c-p*c.sum();expected=p*(utility-(p*utility).sum())
    torch.testing.assert_close(estimated,expected)


def test_zero_credit_exactly_nests_all_parent_gradients_and_forward_values():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(6);parent=SplitEventHeads(payload=4,depth=2)
        candidate=StateCreditEventHeads(payload=4,depth=2,state_credit=0.)
        candidate.load_state_dict(parent.state_dict());row=episodes('order',4,4,2201)[0]
        outputs=[]
        for model in (parent,candidate):
            torch.manual_seed(311);model.train();z,y,state,_=predict(model,row)
            F.cross_entropy(z,y).backward();outputs.append(z.detach())
        torch.testing.assert_close(*outputs,rtol=0,atol=0)
        for a,b in zip(parent.parameters(),candidate.parameters()):
            if a.grad is None:assert b.grad is None
            else:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


def test_added_credit_preserves_hard_forward_inference_rng_and_activity():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(6);model=StateCreditEventHeads(payload=4,depth=2)
        row=episodes('order',4,4,2201)[0];outputs=[];states=[];rng=[]
        for training in (False,True):
            torch.manual_seed(311);model.train(training)
            z,y,state,_=predict(model,row);outputs.append(z.detach());states.append(state);rng.append(torch.get_rng_state())
            if training:
                F.cross_entropy(z,y).backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.testing.assert_close(*outputs,rtol=0,atol=0)
        assert torch.equal(*rng)
        assert states[0].selected_updates==states[1].selected_updates==16*4
        assert states[0].candidate_scores==states[1].candidate_scores==16*8
        assert not states[0].credit_memories
        assert len(states[1].credit_memories)==4*2*2*2
        for k,v in states[0].memories.items():torch.testing.assert_close(v,states[1].memories[k],rtol=0,atol=0)
        saved=copy.deepcopy(states[1].detach());saved.clear_source(0)
        assert all(k[2]!=0 for k in saved.credit_memories)


def test_full_depth_shared_protected_paired_timing_keeps_forward_and_credit():
    from experiments.paired_timing_tasks import paired_timing_episodes
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(611)
        model=StateCreditEventHeads(payload=8,depth=8,shared_maps=True,protected_pairs=2,classes=2)
        row=paired_timing_episodes(4,8,2201)[0];outputs=[];states=[]
        for training in (False,True):
            torch.manual_seed(319);model.train(training)
            z,y,state,_=predict(model,row);outputs.append(z.detach());states.append(state)
            if training:
                F.cross_entropy(z,y).backward()
                for layer in model.queries:
                    for query in layer:
                        assert query.weight.grad is not None
                        assert bool(query.weight.grad.abs().sum()>0)
                        assert bool(torch.isfinite(query.weight.grad).all())
        torch.testing.assert_close(*outputs,rtol=0,atol=0)
        assert states[0].selected_updates==states[1].selected_updates==16*16
        assert states[0].candidate_scores==states[1].candidate_scores==16*32
        assert not states[0].credit_memories
        assert len(states[1].credit_memories)==4*8*2*2
