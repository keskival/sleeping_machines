"""Winner-only value differentiation without removing routing counterfactuals.

Run only through a unique one-job safe queue. No fitted weights are updated.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import RaceLayer, SharedEventModel
from e117_serial_event_shd import batch, load_items


def gradient_error(first, second):
    a = torch.cat([v.reshape(-1) for v in first])
    b = torch.cat([v.reshape(-1) for v in second])
    return {"max_absolute": float((a-b).abs().max()),
            "relative_l2": float((a-b).norm()/a.norm().clamp_min(1e-30))}


def layer_contracts():
    rows = []
    for context, empty_options, options in ((False, False, 3), (True, False, 3),
                                           (True, True, 3), (False, False, 7)):
        torch.manual_seed(127)
        full = RaceLayer(8, 8, 1., True, options=options, memory_backend="linear",
                         global_context=context).double()
        if context:
            with torch.no_grad():
                full.bridge_value.normal_(0,.02); full.bridge_route.normal_(0,.1)
        if empty_options:
            with torch.no_grad():
                full.route.zero_(); full.route_bias.fill_(-100); full.route_bias[0]=100
        winner = copy.deepcopy(full); winner.value_backward="winner"
        x = torch.randn(23,8,dtype=torch.float64)
        t = torch.arange(23,dtype=torch.float64)*.03
        c = torch.rand(23,dtype=torch.float64)+1
        keys = torch.arange(23)%3
        target = torch.randn(23,8,dtype=torch.float64)
        clock_teacher = torch.randn(23,dtype=torch.float64)
        outputs, gradients, stats = [], [], []
        for net in (full,winner):
            inputs=[v.clone().requires_grad_() for v in (x,t,c)]
            y,clock,st,_=net(*inputs,keys,global_keys=torch.zeros(23,dtype=torch.long) if context else None)
            loss=(y*target).sum()+(clock*clock_teacher).sum()
            grad=torch.autograd.grad(loss,[*inputs,*net.parameters()])
            outputs.append((y.detach(),clock.detach()));gradients.append(grad);stats.append(st)
        assert all(torch.equal(a,b) for a,b in zip(*outputs))
        error=gradient_error(*gradients)
        assert error["max_absolute"]<1e-11 and error["relative_l2"]<1e-11,error
        assert stats[0]["value_evaluations"]==23*options
        assert stats[1]["value_evaluations"]==23*(options+1)
        assert stats[1]["differentiable_value_evaluations"]==23
        # Trace consumers can differentiate all returned alternatives.
        winner.zero_grad(set_to_none=True)
        _,_,tr_st,tr=winner(x,t,c,keys,trace=True,
                           global_keys=torch.zeros(23,dtype=torch.long) if context else None)
        assert tr["alternatives"].requires_grad
        assert tr_st["differentiable_value_evaluations"]==23*options
        tr["alternatives"].sum().backward()
        assert bool((winner.value.grad.abs().sum((1,2))>0).all())
        rows.append({"context":context,"empty_losing_options":empty_options,"options":options,
                     "exact_forward":True,"gradient_error":error,"work":stats,
                     "full_trace_graph_preserved":True})
    return rows


def checkpoint_case(path, inputs, repeats):
    saved=torch.load(path,weights_only=False,map_location="cpu")
    config={"global_context_layers":tuple(int(j) for j in saved["args"].get("global_context_layers","").split(",") if j)}
    models={k:SharedEventModel(**config,value_backward=k) for k in ("full","winner")}
    for model in models.values():model.load_state_dict(saved["state_dict"])
    outputs,grads,stats=[],[],[]
    for model in models.values():
        model.train();model.zero_grad(set_to_none=True)
        z,summary,st,_=model(*inputs[:4],len(inputs[-1]))
        F.cross_entropy(z,inputs[-1]).backward()
        outputs.append((z.detach(),summary.detach()))
        grads.append([p.grad.detach().clone() for p in model.parameters()])
        stats.append(st)
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
    error=gradient_error(*grads)
    assert error["relative_l2"]<2e-5,error
    # Ordinary value differentiation changed; the complete router surrogate did not.
    route_errors={n:gradient_error([dict(models['full'].named_parameters())[n].grad],
                                 [dict(models['winner'].named_parameters())[n].grad])
                  for n in dict(models['full'].named_parameters()) if '.route' in n or '.bridge_route' in n}
    for row in route_errors.values():assert row["relative_l2"]<2e-5,row
    times={k:{"forward_s":[],"backward_s":[],"forward_backward_s":[]} for k in models}
    for repeat in range(repeats+1):
        for kind in (("full","winner") if repeat%2 else ("winner","full")):
            model=models[kind];model.train();model.zero_grad(set_to_none=True)
            start=time.perf_counter()
            z=model(*inputs[:4],len(inputs[-1]))[0]
            loss=F.cross_entropy(z,inputs[-1]);middle=time.perf_counter()
            loss.backward();end=time.perf_counter()
            if repeat:
                times[kind]["forward_s"].append(middle-start)
                times[kind]["backward_s"].append(end-middle)
                times[kind]["forward_backward_s"].append(end-start)
    medians={k:{name:float(np.median(v)) for name,v in row.items()} for k,row in times.items()}
    width=models['full'].dim;events=len(inputs[0]);depth=len(models['full'].layers);choices=3
    features=sum(2*width+1+(width+1 if layer.bridge_value is not None else 0)
                 for layer in models['full'].layers)
    # Two contraction VJPs per differentiable map; no memory/route/nonlinearity/optimizer charge here.
    value_flops={"full":6*events*choices*width*features,
                 "winner":2*events*(choices+3)*width*features}
    return {"checkpoint":str(path),"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "config":config,"events":events,"queries":len(inputs[-1]),"depth":depth,
            "exact_training_logits_summary":True,"gradient_error":error,
            "route_gradient_errors":route_errors,"timings":times,"median_timings":medians,
            "forward_backward_speedup":medians['full']['forward_backward_s']/medians['winner']['forward_backward_s'],
            "value_map_forward_backward_flops":value_flops,
            "value_work_boundary":"2 per contraction MAC; all candidate forward + selected recomputation + feature/weight VJPs; excludes other training work",
            "work":{k:{"forward_values":sum(s['value_evaluations'] for s in st['layers']),
                        "differentiable_values":sum(s.get('differentiable_value_evaluations',s['value_evaluations']) for s in st['layers'])}
                    for k,st in zip(models,stats)}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--tag",required=True)
    parser.add_argument("--repeats",type=int,default=9);args=parser.parse_args()
    if Path(args.tag).name!=args.tag or args.repeats<3:raise ValueError("Invalid settings")
    out=Path("experiments/results/e127")/(args.tag+".json");out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    checks=layer_contracts()
    examples=load_items(40,.01,4,"fit_spk",6);inputs=batch(examples)
    cases=[checkpoint_case(Path(path),inputs,args.repeats) for path in (
        "experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt",
        "experiments/results/e122/d8_n4096_bridge_frozen_s6.pt")]
    result={"status":"completed","contracts":checks,"cases":cases,
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (Path(__file__),Path('sleeping_machines/shared_event.py'),Path('sleeping_machines/event_memory.py'))},
            "hardware":{"platform":platform.platform(),"torch":torch.__version__,"threads":1,"device":"cpu"},
            "max_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "energy_joules":None,"scope":"First-order backend equivalence and alternating one-thread timing; no fitting updates, optimizer or energy measurement"}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"contracts":len(checks),"cases":[{k:r[k] for k in
        ('checkpoint','gradient_error','forward_backward_speedup','value_map_forward_backward_flops','median_timings')} for r in cases]}),flush=True)


if __name__=="__main__":main()
