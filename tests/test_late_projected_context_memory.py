import copy
import torch
from sleeping_machines.context_addressed_memory import ContextAddressedNativeModel
from sleeping_machines.late_projected_context_memory import LateProjectedContextModel


def pair():
    torch.manual_seed(731)
    old=ContextAddressedNativeModel(payload=4,depth=2,heads=2,pool=2,order=1,buckets=64).double()
    new=LateProjectedContextModel(payload=4,depth=2,heads=2,pool=2,order=1,buckets=64).double()
    new.load_state_dict(old.state_dict())
    with torch.no_grad():
        old.memory_read_map.weight.normal_(0,.03)
        new.memory_read_map.weight.copy_(old.memory_read_map.weight)
    return old,new


def test_frozen_projection_placement_preserves_outputs_and_whole_graph_gradients():
    old,new=pair();tokens=torch.tensor([1,2,1,3,1,2,1,3,1,2,1,3])
    torch.manual_seed(51);z,_=old.forward_chunk(tokens)
    torch.manual_seed(51);q,_=new.forward_chunk(tokens)
    torch.testing.assert_close(z,q,atol=2e-10,rtol=2e-10)
    z.square().mean().backward();q.square().mean().backward()
    for (name,p),(_,r) in zip(old.named_parameters(),new.named_parameters()):
        if p.grad is None:assert r.grad is None or r.grad.abs().sum()==0,name
        else:torch.testing.assert_close(p.grad,r.grad,atol=2e-9,rtol=2e-8,msg=name)


def test_detached_retrieval_restores_projection_credit_only_for_late_variant():
    old,new=pair();prefix=torch.tensor([1,2,1,3,1,2,1,3])
    norms=[]
    for model in (old,new):
        torch.manual_seed(61)
        with torch.no_grad():_,state=model.forward_chunk(prefix)
        state.detach();z,_=model.forward_chunk(torch.tensor([1]),state)
        z.square().mean().backward();grad=model.memory_write_map.weight.grad
        norms.append(0. if grad is None else float(grad.norm()))
    assert norms[0]==0 and norms[1]>0


def test_current_projection_updates_old_features_and_linear_adjoint_is_exact():
    _,model=pair();state=model.new_state();address=model.address([1]);feature=torch.arange(8,dtype=torch.float64)/8
    state.slots[address]=3*feature;state.counts[address]=3
    value=model.read_value(state,address);before=value.detach().clone()
    adjoint=torch.linspace(-.3,.5,8,dtype=torch.float64)
    (value@adjoint).backward()
    torch.testing.assert_close(model.memory_write_map.weight.grad,adjoint[:,None]*feature[None,:],rtol=0,atol=0)
    with torch.no_grad():model.memory_write_map.weight.add_(.1*torch.eye(8,dtype=torch.float64))
    torch.testing.assert_close(model.read_value(state,address)-before,.1*feature,atol=1e-15,rtol=1e-13)


def test_projected_slots_cannot_be_loaded_as_raw_feature_state():
    old,new=pair()
    try:new.forward_chunk(torch.tensor([1]),old.new_state())
    except TypeError:pass
    else:raise AssertionError('Memory coordinate schema must not be silently mixed')
