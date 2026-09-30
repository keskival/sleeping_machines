"""Matched initialization, local frequency teacher and realized phase credit."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from e139_fine_packet_model import FinePacketModel, load_marked, marked_batch
from e140_phase_packet_model import PhasePacketModel, phase_optimizer
from sleeping_machines.event_memory import linear_memory
from sleeping_machines.rotating_memory import rotating_memory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    out = Path("experiments/results/e140")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError("Unique output required")
    torch.set_num_threads(1)
    torch.manual_seed(140)
    dtype = torch.float64
    x = torch.randn(9, 4, dtype=dtype)
    t = torch.tensor([.01,.03,.09,.1,.12,.18,.3,.4,.5], dtype=dtype)
    c = torch.arange(1, 10, dtype=dtype)
    keys = torch.tensor([0,0,1,0,1,0,1,0,1])
    taus = torch.tensor([.02,.2], dtype=dtype)
    phase = torch.zeros(2,2, dtype=dtype, requires_grad=True)
    old, mass, work = linear_memory(x,t,c,keys,taus)
    new, new_mass, new_work = rotating_memory(x,t,c,keys,taus,phase)
    assert torch.equal(new,old) and torch.equal(new_mass,mass) and new_work == work
    teacher = torch.randn_like(new)
    gradient = torch.autograd.grad((new*teacher).sum(),phase)[0]
    predicted = torch.zeros_like(phase)
    for i in range(len(x)):
        previous = (keys == keys[i]) & (torch.arange(len(x)) <= i)
        lag = t[i]-t[previous]
        weights = c[previous,None]*torch.exp(-lag[:,None]/taus)
        lag_mean = torch.einsum("nk,nd->kd",weights*lag[:,None]/taus,x[previous])/(mass[i,:,None]+1e-4)
        pairs = lag_mean.reshape(2,2,2)
        jlag = torch.stack((-pairs[...,1],pairs[...,0]),-1)
        predicted += (teacher[i].reshape(2,2,2)*jlag).sum(-1)
    teacher_error = float((gradient-predicted).abs().max())
    assert teacher_error < 1e-12 and gradient.norm() > 0
    nonzero = torch.tensor([[.2,-.3],[.7,-.4]], dtype=dtype)
    fast, _, _ = rotating_memory(x,t,c,keys,taus,nonzero)
    serial, _, _ = rotating_memory(x,t,c,keys,taus,nonzero,sequential=True)
    serial_error = float((fast-serial).abs().max())
    assert serial_error < 1e-12
    parent_path = Path("experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt")
    parent = torch.load(parent_path, weights_only=False, map_location="cpu")
    base, model = FinePacketModel(parent), PhasePacketModel(parent)
    rows = load_marked(8,"fit_spk",6)
    packed = marked_batch(rows[:4])
    base.eval(); model.eval()
    with torch.no_grad():
        old_logits,_,_,old_trace=base(packed,trace=True)
        logits,_,stats,trace=model(packed,trace=True)
    initial_error = float((old_logits-logits).abs().max())
    assert torch.equal(old_logits,logits)
    for before,after in zip(old_trace,trace):
        assert torch.equal(before["winner"],after["winner"])
        assert torch.equal(before["times"],after["times"])
    optimizer = phase_optimizer(model,parent,.00007,.0003,.003)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    loss = F.cross_entropy(model(packed)[0],packed[4]); loss.backward()
    phase_norms = [float(l.memory_phase.grad.norm()) for l in model.core.layers]
    assert min(phase_norms) > 0
    total_norm = sum(v*v for v in phase_norms)**.5
    original = [l.memory_phase.detach().clone() for l in model.core.layers]
    with torch.no_grad():
        for layer in model.core.layers:
            layer.memory_phase.add_(layer.memory_phase.grad,alpha=-1e-5/total_norm)
        after = F.cross_entropy(model(packed)[0],packed[4])
        for layer,snapshot in zip(model.core.layers,original):
            layer.memory_phase.copy_(snapshot)
    result={"status":"completed","zero_phase_initial_logits_error":initial_error,
        "zero_phase_races_and_clocks_match":True,"primitive_phase_teacher_error":teacher_error,
        "primitive_parallel_serial_error":serial_error,"primitive_phase_gradient_norm":float(gradient.norm()),
        "phase_gradient_norm_per_layer":phase_norms,"fit_probe_loss_change":float(after-loss.detach()),
        "predicted_loss_change":-1e-5*total_norm,"probe_step_l2":1e-5,
        "phase_parameters":sum(l.memory_phase.numel() for l in model.core.layers),
        "checkpoint_sha256":hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "source_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__),Path("experiments/e140_phase_packet_model.py"),
             Path("sleeping_machines/rotating_memory.py"),Path("experiments/e139_fine_packet_model.py"))},
        "scope":"Eight fitting utterances; exact source/memory chain rule with the existing route surrogate; no retained probe, test access or benchmark quality claim"}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
