"""Exact factorized query/teacher and matched winning-program initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e137_compact_scattering_model import CompactMemoryHead, CompactScatteringClassifier
from e117_serial_event_shd import batch, load_items


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    out = Path('experiments/results/e137') / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Invalid output')
    torch.set_num_threads(1); torch.manual_seed(137)
    head = CompactMemoryHead(3, 5, 4, 3, 2).double()
    z = torch.randn(7, 23, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(7) % 3
    factored = head(z)
    memory_weight = torch.einsum('cr,rb,rd->cbd', head.class_weight, head.bank, head.channel)
    weight = torch.cat((head.packet.weight, memory_weight.flatten(1)), 1)
    dense = F.linear(z, weight, head.packet.bias)
    parameters = [z, *head.parameters()]
    left = torch.autograd.grad(F.cross_entropy(factored, labels), parameters, retain_graph=True)
    right = torch.autograd.grad(F.cross_entropy(dense, labels), parameters)
    value_error = float((factored - dense).detach().abs().max())
    gradient_error = max(float((x-y).abs().max()) for x,y in zip(left,right))
    g = torch.randn_like(factored)
    teacher = torch.einsum('kr,rb,rd->kbd', g @ head.class_weight, head.bank, head.channel)
    reverse = torch.autograd.grad((head(z) * g).sum(), z)[0][:, 3:].reshape(-1, 5, 4)
    teacher_error = float((reverse - teacher).detach().abs().max())
    assert max(value_error, gradient_error, teacher_error) < 1e-12
    learned = CompactScatteringClassifier(); frozen = CompactScatteringClassifier(angles='frozen')
    items = load_items(40, .01, 4, 'fit_spk', 6)
    learned.calibrate(items, batch); frozen.calibrate(items, batch)
    data = batch(items)
    zl, _, sl, tl = learned(*data[:4], len(items), trace=True)
    zf, _, sf, tf = frozen(*data[:4], len(items), trace=True)
    assert torch.equal(zl, zf)
    assert all(torch.equal(x['winner'],y['winner']) and torch.equal(x['times'],y['times'])
               for x,y in zip(tl,tf))
    F.cross_entropy(zl, data[-1]).backward()
    gradients = learned.angle_weight.grad.flatten(1).norm(dim=1)
    assert bool((gradients > 0).all())
    result = {'status':'completed', 'factorized_dense_max_error':value_error,
              'all_parameter_gradient_max_error':gradient_error,
              'local_state_teacher_max_error':teacher_error,
              'matched_initial_logits_max_error':float((zl-zf).detach().abs().max()),
              'matched_actual_winners_and_clocks':True,
              'twelve_angle_layer_gradient_norms':gradients.tolist(),
              'learned_angle_train_parameters':sum(p.numel() for p in learned.parameters() if p.requires_grad),
              'frozen_angle_train_parameters':sum(p.numel() for p in frozen.parameters() if p.requires_grad),
              'head_parameters':sum(p.numel() for p in learned.head.parameters()),
              'head_forward_macs_per_query':learned.head.forward_macs_per_query(),
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (Path(__file__),Path('experiments/e137_compact_scattering_model.py'))},
              'scope':'Numerical query/teacher contract and four fitting utterances; no classification benchmark claim.'}
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
