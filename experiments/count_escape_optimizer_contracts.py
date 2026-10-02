"""Guarded full-depth optimizer/recovery contracts for the prepared count repair."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
from e120_shared_tasks import text_slice
from sleeping_machines.count_carrying_language import CountCarryingNativeModel, fit_stream_counts
from sleeping_machines.count_escape_gate import GatedCountCarryingNativeModel


def verify_variant(tokens, counts, message, gate):
    with torch.random.fork_rng():
        torch.manual_seed(831)
        model = GatedCountCarryingNativeModel(payload=16, depth=8, heads=2, pool=2,
            orders=4, count_message=message, escape_gate=gate)
        parent = CountCarryingNativeModel(payload=16, depth=8, heads=2, pool=2, orders=4)
        parent.load_state_dict({name:value for name,value in model.state_dict().items() if name in parent.state_dict()})
        for net in (model, parent):
            net.register_stream('fit', counts); net.use_stream('fit'); net.train()
        rng = torch.get_rng_state(); outputs = []; parent_gradients = {}
        for net in (parent, model):
            torch.set_rng_state(rng); z, state = net.forward_chunk(tokens[:16])
            outputs.append(z.detach()); F.cross_entropy(z, tokens[1:17]).backward()
            if net is parent:
                parent_gradients = {n:None if p.grad is None else p.grad.clone() for n,p in net.named_parameters()}
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            if name in parent_gradients:
                expected = parent_gradients[name]
                if expected is None: assert parameter.grad is None
                else: torch.testing.assert_close(parameter.grad, expected, rtol=2e-6, atol=2e-7)
        for module in (model.count_message, model.escape_gate):
            if module is not None:
                assert module.weight.grad is not None and bool(module.weight.grad.abs().sum()>0)
                assert bool(torch.isfinite(module.weight.grad).all())
        for layer in model.queries:
            for query in layer:
                assert query.weight.grad is not None and bool(query.weight.grad.abs().sum()>0)
                assert bool(torch.isfinite(query.weight.grad).all())
        state = state.detach(); optimizer = torch.optim.Adam(model.parameters(), lr=.002)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); optimizer.step()
        buffer = io.BytesIO()
        torch.save(dict(weights=model.state_dict(), optimizer=optimizer.state_dict(),
            state=state, rng=torch.get_rng_state()), buffer)
        buffer.seek(0); saved = torch.load(buffer, weights_only=False)
        recovered = copy.deepcopy(model); recovered.load_state_dict(saved['weights'])
        other = torch.optim.Adam(recovered.parameters(), lr=.002); other.load_state_dict(saved['optimizer'])
        replay = copy.deepcopy(saved['state']); values = []
        for net, opt, live in ((model, optimizer, state), (recovered, other, replay)):
            torch.set_rng_state(saved['rng']); opt.zero_grad(set_to_none=True)
            z, _ = net.forward_chunk(tokens[16:32], live); values.append(z.detach())
            F.cross_entropy(z, tokens[17:33]).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True); opt.step()
        torch.testing.assert_close(*values, rtol=0, atol=0)
        for a,b in zip(model.parameters(), recovered.parameters()):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        return dict(count_message=message, escape_gate=gate, payload=16, depth=8, heads=2,
            exact_zero_repair_parent_forward=True, zero_repair_parent_gradients_match=True,
            all_head_query_gradients=True, all_enabled_repair_gradients=True,
            exact_next_optimizer_update_recovery=True, persistent_count_position_recovery=True,
            numerical_optimizer_steps_per_training_copy=2,
            scope='Full integrated model numerical prerequisite; not fitted quality or deep-feature evidence')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--tag',required=True)
    args=parser.parse_args(); path=ROOT/'experiments/results/diagnostics'/(args.tag+'.json')
    if Path(args.tag).name!=args.tag or path.exists(): raise ValueError('Unused plain tag required')
    torch.set_num_threads(1); started=time.perf_counter(); tokens=torch.tensor(text_slice(0,2048))
    counts=fit_stream_counts(tokens.numpy(),4)
    variants=[verify_variant(tokens,counts,message,gate) for message,gate in ((True,True),(True,False),(False,True))]
    names=['experiments/count_escape_optimizer_contracts.py','experiments/e120_shared_tasks.py',
        'sleeping_machines/count_carrying_language.py','sleeping_machines/count_escape_gate.py',
        'sleeping_machines/native_stream_language.py','sleeping_machines/addressed_event_heads.py',
        'sleeping_machines/parallel_head_race_language.py','sleeping_machines/sparse_race_language.py',
        'sleeping_machines/parallel_stream_language.py','sleeping_machines/rotating_memory.py']
    result=dict(status='completed',args=vars(args),kind='numerical_optimizer_prerequisites',variants=variants,
        hardware=dict(host=__import__('os').uname().nodename,torch=torch.__version__,device='cpu',threads=1),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        scope='No official-test access or benchmark score. Contract optimizer work is distinct from future fitted-model work.')
    path.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(dict(completed=args.tag,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__': main()
