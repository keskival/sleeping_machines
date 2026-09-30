"""Function-preserving six-block temporal residual on the strongest SHD model.

The old event classifier is frozen. A generic full-source temporal encoder learns
an additive logit correction; its zero head exactly preserves the old answers.
This is a larger, parallel residual architecture, not a replacement accuracy
claim for the original eight-layer core or a reproduction of EventSSM/S7.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from e139_fine_packet_model import augment_marked,load_marked,marked_batch
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.event_state import CoalescedEventStateEncoder


def event_batch(rows,window):
    sources,times,assignments,closures,receivers,counts=[],[],[],[],[],[]
    offset=0
    for receiver,row in enumerate(rows):
        raw=row[7]
        bins,assignment,count=np.unique(np.floor(raw/window).astype(np.int64),
            return_inverse=True,return_counts=True)
        sources.append(row[0][row[5]]*18+row[6])
        times.append(raw)
        assignments.append(assignment+offset)
        closures.append(((bins+1)*window).astype(np.float32))
        counts.append(count.astype(np.float32))
        receivers.append(np.full(len(bins),receiver,np.int64))
        offset+=len(bins)
    return tuple(torch.from_numpy(np.concatenate(x)) for x in
        (sources,times,assignments,closures,receivers,counts))


def scores_for(core,encoder,rows,window):
    packed=marked_batch(rows)
    with torch.no_grad():
        original,_,old_stats,_=core(*packed[:4],len(rows))
    scores,_,_,new_stats=encoder(*event_batch(rows,window),len(rows))
    return original+scores,packed[4],old_stats,new_stats,original


@torch.no_grad()
def evaluate(core,encoder,items,bs,window):
    encoder.eval()
    correct,nll,old_correct,old_nll=0,0.,0,0.
    predictions,labels=[],[]
    for start in range(0,len(items),bs):
        score,y,_,_,parent=scores_for(core,encoder,items[start:start+bs],window)
        if not torch.isfinite(score).all():raise FloatingPointError("Nonfinite logits")
        pred=score.argmax(-1)
        correct+=int((pred==y).sum());nll+=float(F.cross_entropy(score,y,reduction="sum"))
        old_correct+=int((parent.argmax(-1)==y).sum())
        old_nll+=float(F.cross_entropy(parent,y,reduction="sum"))
        predictions.extend(pred.tolist());labels.extend(y.tolist())
    return dict(accuracy=correct/len(items),correct=correct,n=len(items),nll=nll/len(items),
        predictions=predictions,labels=labels,parent_correct=old_correct,parent_nll=old_nll/len(items))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--tag",required=True)
    p.add_argument("--checkpoint",default="experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt")
    p.add_argument("--limit",type=int,default=6144)
    p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--bs",type=int,default=4)
    p.add_argument("--width",type=int,default=128)
    p.add_argument("--modes",type=int,default=64)
    p.add_argument("--depth",type=int,default=6)
    p.add_argument("--window",type=float,default=.01)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--seed",type=int,default=6)
    p.add_argument("--calibrate",action="store_true")
    a=p.parse_args()
    out=Path("experiments/results/e143")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists() or min(a.limit,a.epochs,a.bs)<1 or a.window<=0:
        raise ValueError("Unique output and positive settings required")
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    started=time.perf_counter()
    parent=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    core=SharedEventModel(depth=8,memory_backend="linear")
    core.load_state_dict(parent["state_dict"]);core.requires_grad_(False);core.eval()
    encoder=CoalescedEventStateEncoder(sources=720,width=a.width,modes=a.modes,depth=a.depth)
    torch.nn.init.zeros_(encoder.head.weight);torch.nn.init.zeros_(encoder.head.bias)
    opt=torch.optim.Adam(encoder.parameters(),lr=a.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.lr/10)
    fit=load_marked(a.limit,"fit_spk",a.seed)
    held=load_marked(8 if a.calibrate else 512,"val_spk",a.seed+1)
    if len(fit)!=a.limit or len(held)!=(8 if a.calibrate else 512):raise ValueError("Unavailable split")
    if set(r[8] for r in fit)&set(r[8] for r in held):raise ValueError("Overlapping splits")
    order_rng=np.random.default_rng(a.seed+2)
    if not a.calibrate:order_rng.bit_generator.state=parent.get("numpy_rng",parent.get("order_rng"))
    aug_rng=np.random.default_rng(a.seed+122)
    if not a.calibrate and "augmentation_rng" in parent:aug_rng.bit_generator.state=parent["augmentation_rng"]
    paths=[Path(__file__),Path("sleeping_machines/event_state.py"),
        Path("sleeping_machines/event_memory.py"),Path("sleeping_machines/rotating_memory.py"),
        Path("sleeping_machines/shared_event.py"),Path("experiments/e139_fine_packet_model.py"),
        Path("experiments/e117_serial_event_shd.py"),Path("experiments/e51_shd_world.py")]
    result=dict(status="running",args=vars(a),curve=[],
        source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in paths},
        checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
        fit_absolute_ids=[r[8] for r in fit],dev_absolute_ids=[r[8] for r in held],
        old_parameters=sum(p.numel() for p in core.parameters()),
        new_parameters=sum(p.numel() for p in encoder.parameters()),
        architecture="Frozen generic D8 width32 carrier plus parallel trainable D6 width128 signed event-state encoder; zero logit head preserves parent",
        packetization="Learned 720 source-address table; 700 original addresses injective. Nonempty 10ms global closures; exact first affine raw-impulse endpoints, nonlinear outputs only at closures",
        clock_credit="Only winning clocks emit; local zero-forward candidate-delay surrogate. Last emission time is unobserved by completed-query CE",
        protocol="Exploratory train-file speaker3/6 development; no official test; inherited trained parent; independent residual learner, not from scratch or an EventSSM reproduction",
        hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1),energy_joules=None)
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        temp=out.with_suffix(".json.tmp");temp.write_text(json.dumps(result,indent=2)+"\n");temp.replace(out)
    initial=evaluate(core,encoder,held,a.bs,a.window)
    assert initial["correct"]==initial["parent_correct"] and initial["nll"]==initial["parent_nll"]
    result["initial"]=dict(dev=initial)
    if not a.calibrate and initial["correct"]!=370:raise ValueError("Unexpected parent/split predictions")
    persist();print(json.dumps({"initial_dev":initial["accuracy"],"new_parameters":result["new_parameters"]}),flush=True)
    for epoch in range(1,a.epochs+1):
        encoder.train();order=order_rng.permutation(len(fit));train_start=time.perf_counter()
        acc=dict(nll=0.,steps=0,clipped=0,source_events=0,old_packets=0,new_packets=0,
            source_table_projection_macs=0,source_transported_scalars=0,source_payload_sum_scalars=0,
            event_projection_macs=0,gate_macs=0,state_compositions=0,clock_candidates=0,
            layer_gradient_norm=[0.]*a.depth)
        for start in range(0,len(fit),a.bs):
            rows=[augment_marked(fit[j],aug_rng) for j in order[start:start+a.bs]]
            opt.zero_grad(set_to_none=True)
            score,y,old,new,_=scores_for(core,encoder,rows,a.window)
            loss=F.cross_entropy(score,y)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite loss")
            loss.backward()
            for i,layer in enumerate(encoder.layers):
                acc["layer_gradient_norm"][i]+=float(torch.sqrt(sum((p.grad.square().sum() for p in layer.parameters() if p.grad is not None),score.new_tensor(0.))))
            norm=torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.,error_if_nonfinite=True)
            acc["clipped"]+=int(norm>1);opt.step()
            acc["nll"]+=float(loss.detach())*len(rows);acc["steps"]+=1
            acc["source_events"]+=new[0]["source_events"];acc["old_packets"]+=old["packets"]
            acc["new_packets"]+=new[0]["emitted_vectors"]
            for name in ("source_table_projection_macs","source_transported_scalars","source_payload_sum_scalars"):
                acc[name]+=new[0][name]
            acc["event_projection_macs"]+=sum(r.get("input_projection_macs",0)+r["output_projection_macs"] for r in new)
            acc["gate_macs"]+=sum(r["nonlinear_gate_macs"] for r in new)
            acc["state_compositions"]+=sum(r["state_scan_compositions"] for r in new)
            acc["clock_candidates"]+=sum(r["clock_candidates"] for r in new)
            if acc["steps"]%128==0:
                result["progress"]=dict(epoch=epoch,completed_examples=start+len(rows),online_nll=acc["nll"]/(start+len(rows)))
                persist();print(json.dumps(result["progress"]),flush=True)
        acc.update(online_nll=acc["nll"]/len(fit),training_wall_s=time.perf_counter()-train_start)
        encoder.eval()
        row=dict(epoch=epoch,training=acc,fit=evaluate(core,encoder,fit,a.bs,a.window),
            dev=evaluate(core,encoder,held,a.bs,a.window),lr=opt.param_groups[0]["lr"])
        assert all(torch.equal(v,core.state_dict()[k]) for k,v in parent["state_dict"].items())
        row["parent_state_unchanged"]=True
        result["curve"].append(row);result["final"]=row;scheduler.step();persist()
        torch.save(dict(encoder_state_dict=encoder.state_dict(),optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(),args=vars(a),result=result,order_rng=order_rng.bit_generator.state,
            augmentation_rng=aug_rng.bit_generator.state,torch_rng=torch.get_rng_state()),out.with_suffix(".pt"))
        print(json.dumps({"epoch":epoch,"fit":row["fit"]["accuracy"],"dev":row["dev"]["accuracy"],
            "training_wall_s":acc["training_wall_s"],"rss_kb":result["max_rss_kb"]}),flush=True)
    result["status"]="completed";persist()


if __name__=="__main__":main()
