"""Small dense controls on the consolidated model's exact synthetic examples.

These are targeted controls, not a hyperparameter search. Width/depth 32/2,
same neural schedule and fitting budget. Recall additionally exposes the dense
learner to the same 4,000 examples that trained the shared pointer memory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from e120_shared_tasks import Example, modular, recall, prefix
from e61_race_attention import make_perm, sample
from e36_transformer import EventTransformer
from e120_shared_bench import loss_for, BUILDERS


class EventLSTM(nn.Module):
    def __init__(self, bands, classes, dim, depth):
        super().__init__()
        self.embedding=nn.Embedding(bands,dim)
        self.time_map=nn.Linear(4,dim,bias=False)
        self.rnn=nn.LSTM(dim,dim,depth,batch_first=True)
        self.head=nn.Linear(dim,classes)
        self.register_buffer("tau",torch.tensor([.05,.2,.8]))
    def forward(self,b,t,mask):
        features=torch.cat((t.new_ones((*t.shape,1)),torch.exp(-t[:,:,None]/self.tau)),dim=-1)
        x=self.embedding(b)+self.time_map(features)
        packed=pack_padded_sequence(x,(~mask).sum(1).cpu(),batch_first=True,enforce_sorted=False)
        _,(hidden,_)=self.rnn(packed)
        return self.head(hidden[-1])


class CountedTransformer(EventTransformer):
    """Expose the count mark available to the common gesture encoder."""
    def __init__(self,bands,classes,dim,depth):
        super().__init__(bands,classes,dim,depth,1.2)
        self.count_map=nn.Linear(1,dim,bias=False)
        nn.init.zeros_(self.count_map.weight)
    def forward(self,b,t,mask,counts):
        angle=t[:,:,None]*self.freq
        x=self.ch(b)+torch.cat((angle.sin(),angle.cos()),-1)
        x=x+self.count_map(torch.log1p(counts[:,:,None])/10)
        x=self.enc(x,src_key_padding_mask=mask)
        x=x.masked_fill(mask[:,:,None],0).sum(1)/(~mask).sum(1,keepdim=True)
        return self.out(x)


def predict(net,b,t,mask,rows):
    if not isinstance(net,CountedTransformer):return net(b,t,mask)
    counts=torch.zeros_like(t)
    for i,row in enumerate(rows):
        counts[i,:len(row.prefix.counts)]=torch.from_numpy(row.prefix.counts.copy()).float()
    return net(b,t,mask,counts)


def batch(rows):
    lengths=[len(r.prefix.channels) for r in rows]
    b=torch.zeros(len(rows),max(lengths),dtype=torch.long)
    t=torch.zeros_like(b,dtype=torch.float32)
    mask=torch.ones_like(b,dtype=torch.bool)
    for i,r in enumerate(rows):
        n=lengths[i];b[i,:n]=torch.from_numpy(r.prefix.channels.copy())
        t[i,:n]=torch.from_numpy(r.prefix.times.copy());mask[i,:n]=False
    return b,t,mask,torch.tensor([r.label for r in rows])


@torch.no_grad()
def evaluate(net,rows,bs):
    net.eval();total=0.;pred=[]
    for start in range(0,len(rows),bs):
        part=rows[start:start+bs]
        b,t,mask,y=batch(part);z=predict(net,b,t,mask,part)
        pred+=z.argmax(-1).tolist()
        total+=float(loss_for(z,rows[start:start+bs],"sum"))
    correct=sum(p==r.label for p,r in zip(pred,rows))
    result={"n":len(rows),"nll":total/len(rows)}
    if rows[0].exposure is None:
        result.update(correct=correct,accuracy=correct/len(rows),predictions=pred)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=tuple(BUILDERS),required=True)
    p.add_argument("--model",choices=("lstm","transformer"),required=True)
    p.add_argument("--tag",required=True);p.add_argument("--epochs",type=int,required=True)
    p.add_argument("--bs",type=int,default=64);p.add_argument("--seed",type=int,default=6)
    p.add_argument("--fit",type=int);p.add_argument("--dev",type=int,default=256)
    a=p.parse_args()
    out=Path("experiments/results/e123")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    if a.task=="modular":
        task=modular(1473,3440,a.seed)
    elif a.task=="recall":
        task=recall(512,256,a.seed)
    else:
        if a.fit is None:raise ValueError("A reference task needs --fit")
        task=BUILDERS[a.task](a.fit,a.dev,a.seed)
    if a.task == "modular":
        task.protocol["note"] = "Same position-tagged triples as E121; all unseen tuples; 200-epoch dense control with no supplied phase primitive"
    fit=list(task.fit)
    if a.task=="recall":
        rng=np.random.default_rng(a.seed);perm=make_perm(32,rng)
        assert np.array_equal(perm,task.protocol["permutation"])
        memory_rows=[]
        for j in range(4000):
            seq,q,y=sample(32,8,perm,rng)
            memory_rows.append(Example(prefix(np.r_[seq,64+q]),y,f"memory:{j}"))
        fit=memory_rows+fit
    dim,depth=32,2
    net=(CountedTransformer(task.config["bands"],task.config["classes"],dim,depth) if a.task=="dvs" and a.model=="transformer" else
         EventLSTM(task.config["bands"],task.config["classes"],dim,depth) if a.model=="lstm" else
         EventTransformer(task.config["bands"],task.config["classes"],dim,depth,1.2))
    opt=torch.optim.Adam(net.parameters(),lr=.003)
    rng=np.random.default_rng(a.seed+10);started=time.perf_counter()
    result={"status":"running","args":vars(a),"dim":dim,"depth":depth,
            "parameters":sum(p.numel() for p in net.parameters()),"protocol":task.protocol,
            "fit_ids":[r.identity for r in fit],"dev_ids":[r.identity for r in task.dev],
            "curve":[],"hardware":{"platform":platform.platform(),"threads":1,"torch":torch.__version__},
            "scope":"Same adapters, labels and seed as the shared model; one small dense setting, not a tuned best baseline; language/market reference has no frozen evidence bank",
            "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(__file__),Path("experiments/e120_shared_tasks.py"),Path("experiments/e36_transformer.py")]},
            "energy_joules":None}
    def persist():
        result.update(wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp=out.with_suffix(".json.tmp");tmp.write_text(json.dumps(result,indent=2)+"\n");tmp.replace(out)
    persist()
    for epoch in range(1,a.epochs+1):
        denominator=max(a.epochs-1,1) if a.task in ("modular","recall") else a.epochs
        lr=.003*(.1+.9*(1+math.cos(math.pi*(epoch-1)/denominator))/2)
        for g in opt.param_groups:g["lr"]=lr
        net.train();total=0.;order=rng.permutation(len(fit))
        for start in range(0,len(fit),a.bs):
            rows=[fit[i] for i in order[start:start+a.bs]]
            b,t,mask,y=batch(rows);z=predict(net,b,t,mask,rows)
            loss=loss_for(z,rows)
            if not torch.isfinite(loss):raise FloatingPointError("Nonfinite baseline loss")
            opt.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
            total+=float(loss.detach())*len(rows)
        if epoch in (1,8) or epoch%10==0 or epoch==a.epochs:
            row={"epoch":epoch,"online_nll":total/len(fit),"fit":evaluate(net,fit,a.bs),
                 "dev":evaluate(net,task.dev,a.bs)}
            result["curve"].append(row);persist()
            print(json.dumps({"epoch":epoch,"fit":row["fit"].get("accuracy",row["fit"]["nll"]),"dev":row["dev"].get("accuracy",row["dev"]["nll"]),
                              "wall_s":result["wall_s"]}),flush=True)
    result.update(status="completed",final=result["curve"][-1],
                  extra={k:evaluate(net,v,a.bs) for k,v in task.extra.items()},
                  fitting_presentations=len(fit)*a.epochs)
    persist()
    torch.save({"state_dict":net.state_dict(),"args":vars(a)},out.with_suffix(".pt"))


if __name__=="__main__":main()
