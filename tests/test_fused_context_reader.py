import torch
from sleeping_machines.fused_context_reader import compile_reader
from sleeping_machines.late_projected_context_memory import LateProjectedContextModel


def test_compilation_preserves_nonzero_reader_predictions_and_raw_slot_state():
    torch.manual_seed(19)
    model=LateProjectedContextModel(payload=4,depth=2,heads=2,order=1,buckets=64).double().eval()
    with torch.no_grad():model.memory_read_map.weight.normal_(0,.1)
    rng=torch.get_rng_state();fused=compile_reader(model)
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    tokens=torch.tensor([1,2,1,3,1,2,1,3]*8)
    def run(m):
        torch.manual_seed(7);state=m.new_state();out=[]
        with torch.no_grad():
            for start in range(0,len(tokens),16):
                z,state=m.forward_chunk(tokens[start:start+16],state);out.append(z);state.detach()
        return torch.cat(out),state
    z,original=run(model);q,compiled=run(fused)
    torch.testing.assert_close(z,q,rtol=2e-10,atol=2e-10)
    for key,value in original.slots.items():torch.testing.assert_close(value,compiled.slots[key],rtol=2e-10,atol=2e-10)
    assert original.counts==compiled.counts
    assert sum(p.numel() for p in model.parameters())-sum(p.numel() for p in fused.parameters())==64


def test_compiled_reader_refuses_training_or_graph_construction():
    model=LateProjectedContextModel(payload=2,depth=1,heads=2)
    fused=compile_reader(model)
    for action in (lambda:fused.train(),lambda:fused.forward_chunk(torch.tensor([1]))):
        try:action()
        except (ValueError,RuntimeError):pass
        else:raise AssertionError('Inference compilation must not silently change fitting parameterization')
