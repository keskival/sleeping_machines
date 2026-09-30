"""Guarded causality, precise-clock, gradient and throughput checks; no quality claim."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleeping_machines.parallel_stream_language import ParallelEventLanguageModel
from sleeping_machines.stream_language import StreamingEventLanguageModel
from sleeping_machines.operation_audit import OperationAudit
from sleeping_machines.language_memory import PROFILES, initialize_language_memory
from sleeping_machines.selective_stream_language import SelectiveEventLanguageModel


def serial(model, tokens, state):
    return torch.stack([model.consume(t, state) for t in tokens]), state


def error(a, b):
    return float((a-b).detach().abs().max())


def timed_step(base, tokens, parallel):
    model = copy.deepcopy(base).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    start = time.perf_counter()
    logits, _ = (model.forward_chunk if parallel else lambda t, s: serial(model, t, s))(
        tokens[:-1], model.new_state())
    loss = F.cross_entropy(logits, tokens[1:])
    forward_s = time.perf_counter()-start
    start = time.perf_counter(); loss.backward(); backward_s = time.perf_counter()-start
    start = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    update_s = time.perf_counter()-start
    return dict(forward_s=forward_s, backward_s=backward_s, update_s=update_s,
                total_s=forward_s+backward_s+update_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--memory-profile',choices=PROFILES,default='inherited')
    parser.add_argument('--content-memory',action='store_true')
    args = parser.parse_args()
    output = ROOT/'experiments/results/parallel_language'/f'{args.tag}.json'
    if Path(args.tag).name != args.tag or output.exists():
        raise ValueError('A unique plain tag is required')
    torch.set_num_threads(1); torch.manual_seed(6)
    start = time.perf_counter()
    model_class = SelectiveEventLanguageModel if args.content_memory else ParallelEventLanguageModel
    model = model_class(width=args.width, modes=args.width//2, depth=6)
    initialize_language_memory(model,args.memory_profile)
    identity_error = None
    if args.content_memory:
        plain = ParallelEventLanguageModel(width=args.width,modes=args.width//2,depth=6)
        plain.load_state_dict({name:value for name,value in model.state_dict().items()
                               if '.memory_control.' not in name})
        sample = torch.tensor([1,3,5,3,8,0,4,2])
        initial_plain,_ = plain.forward_chunk(sample)
        initial_selective,_ = model.forward_chunk(sample)
        torch.testing.assert_close(initial_selective,initial_plain,rtol=3e-4,atol=3e-5)
        identity_error = error(initial_selective,initial_plain)
    with torch.no_grad():
        for layer in model.layers:
            layer.clock.weight.normal_(0, .05); layer.clock.bias.normal_(0, .1)
            if args.content_memory:
                layer.memory_control.weight.normal_(0,.05)
                layer.memory_control.bias.normal_(0,.1)
    tokens = torch.randint(0, 27, (48,))
    cases = []
    for position in (0, 10_000_000):
        reference, candidate = copy.deepcopy(model), copy.deepcopy(model)
        a, b = reference.new_state(), candidate.new_state()
        a.position = b.position = position
        with torch.no_grad():
            _, a = serial(reference, tokens[:7], a)
            _, b = serial(candidate, tokens[:7], b)
        a, b = a.detach(), b.detach()
        reference_logits, a = serial(reference, tokens[7:-1], a)
        candidate_logits, b = candidate.forward_chunk(tokens[7:-1], b)
        torch.testing.assert_close(candidate_logits, reference_logits, rtol=3e-4, atol=3e-5)
        mode_error = max(error(x, y) for x, y in zip(a.modes, b.modes))
        for x, y in zip(a.modes, b.modes):
            torch.testing.assert_close(x, y, rtol=4e-4, atol=1e-4)
        F.cross_entropy(reference_logits, tokens[8:]).backward()
        F.cross_entropy(candidate_logits, tokens[8:]).backward()
        gradient_errors = {}
        for (name, x), (other, y) in zip(reference.named_parameters(), candidate.named_parameters()):
            assert name == other
            if x.grad is None or y.grad is None:
                assert x.grad is None and y.grad is None, name
                continue
            torch.testing.assert_close(y.grad, x.grad, rtol=3e-3, atol=2e-5)
            gradient_errors[name] = error(x.grad, y.grad)
        with torch.no_grad():
            whole, _ = candidate.forward_chunk(tokens[:-1])
            state, chunks = candidate.new_state(), []
            for left, right in ((0,2), (2,9), (9,22), (22,47)):
                logits, state = candidate.forward_chunk(tokens[left:right], state)
                chunks.append(logits)
            chunked = torch.cat(chunks)
            torch.testing.assert_close(chunked, whole, rtol=3e-4, atol=3e-5)
            changed = tokens[:-1].clone(); changed[39:] = (changed[39:]+1)%27
            perturbed, _ = candidate.forward_chunk(changed)
            torch.testing.assert_close(perturbed[:39], whole[:39], rtol=3e-4, atol=3e-5)
        cases.append(dict(position=position, width=args.width, warm_tokens=7,
                          logit_max_abs_error=error(candidate_logits, reference_logits),
                          state_max_abs_error=mode_error, gradient_max_abs_errors=gradient_errors,
                          chunk_max_abs_error=error(chunked, whole),
                          future_perturbation_prefix_error=error(perturbed[:39], whole[:39])))
    with torch.no_grad():
        old = StreamingEventLanguageModel(width=args.width, modes=args.width//2, depth=6)
        old.load_state_dict({name:value for name,value in model.state_dict().items()
                             if '.memory_control.' not in name})
        old_state, precise_state = old.new_state(), model.new_state()
        old_state.position = precise_state.position = 10_000_000
        old.consume(3, old_state); model.consume(3, precise_state)
        old_delay = float(old_state.last_arrival[1])-10_000_000
        precise_delay = float(precise_state.last_arrival[1])-10_000_000
        assert old_delay == 0 and .001 <= precise_delay <= .011
        early_old, _ = old.forward_chunk(tokens[:16])
        early_new, _ = model.forward_chunk(tokens[:16])
        if not args.content_memory:
            torch.testing.assert_close(early_new, early_old, rtol=5e-4, atol=1e-4)
    benchmark_model = model_class(width=256, modes=128, depth=6)
    initialize_language_memory(benchmark_model,args.memory_profile)
    sample = torch.randint(0, 27, (65,))
    speed = {name:[timed_step(benchmark_model, sample, parallel) for _ in range(3)]
             for name, parallel in (('serial_precise',False), ('parallel_precise',True))}
    throughput_ratio = (sum(r['total_s'] for r in speed['serial_precise']) /
                        sum(r['total_s'] for r in speed['parallel_precise']))
    traces = {}
    for name, parallel in (('serial_precise',False), ('parallel_precise',True)):
        probe = copy.deepcopy(benchmark_model).train()
        with OperationAudit() as audit:
            logits, _ = (probe.forward_chunk if parallel else lambda t, s: serial(probe,t,s))(
                sample[:-1], probe.new_state())
            F.cross_entropy(logits,sample[1:]).backward()
        traces[name] = audit.result()
        if not traces[name]['formula_coverage_complete']:
            raise ValueError(traces[name]['unsupported_floating_operators'])
    result = dict(status='completed', args=vars(args), cases=cases,
                  initial_constant_memory_equivalence_error=identity_error,
                  clock_precision=dict(position=10_000_000, legacy_first_delay=old_delay,
                                       corrected_first_delay=precise_delay),
                  speed=speed, measured_step_speedup=throughput_ratio, traces=traces,
                  protocol='Fixed random tokens; no fitting corpus, development selection or official test. Same parameters and payload precision, serial precise-clock reference versus causal parallel scans.',
                  scope='Numerical contracts and one-host implementation throughput, not prediction-quality, training-energy or algorithmic-FLOP supremacy.',
                  hardware=dict(platform=platform.platform(),torch=torch.__version__,device='cpu',threads=1),
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__),ROOT/'sleeping_machines/parallel_stream_language.py',
                                ROOT/'sleeping_machines/stream_language.py',ROOT/'sleeping_machines/event_state.py',
                                ROOT/'sleeping_machines/event_memory.py',ROOT/'sleeping_machines/language_memory.py',
                                ROOT/'sleeping_machines/selective_stream_language.py')},
                  wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],clock_precision=result['clock_precision'],
                          measured_step_speedup=throughput_ratio,wall_s=result['wall_s'])),flush=True)


if __name__ == '__main__':
    main()
