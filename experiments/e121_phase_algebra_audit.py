"""Certify the fitted modular phase rule by circular min/max composition.

Arithmetic structure is used only by this post-training mathematical audit,
never by the model's phase teaching or inference routines.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel


def circular(v, period):
    return (v+period/2)%period-period/2


def cyclic_extrema(a, b):
    """Min and max convolution over a+b mod p; O(p²), not enumeration of triples."""
    p=len(a)
    vals=np.stack([a[j]+b[(np.arange(p)-j)%p] for j in range(p)])
    return vals.min(0), vals.max(0)


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);a=p.parse_args()
    out=Path("experiments/results/e121")/(a.tag+".json")
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    checkpoint=Path("experiments/results/e121/shared_d2_guard_s6_200.pt")
    saved=torch.load(checkpoint,weights_only=False,map_location="cpu")
    model=SharedEventModel(**saved["config"])
    model.load_state_dict(saved["state_dict"],strict=True)
    memory=model.phase_memory
    period=memory.period
    assert period == 17 and len(memory.phase) == 51 and memory.reflection == -1
    codes=memory.phase.numpy().reshape(3,17)
    clocks=memory.centers.numpy()
    x=np.arange(17)
    candidates=[]
    for slope in range(1,17):
        residuals=[];offsets=[]
        for values,sign in zip([*codes,clocks], [1,-1,1,1]):
            raw=values-sign*slope*x
            offset=float(np.angle(np.exp(2j*np.pi*raw/period).mean())*period/(2*np.pi))
            offsets.append(offset);residuals.append(circular(raw-offset,period))
        candidates.append((sum(float(np.max(np.abs(e))) for e in residuals),slope,offsets,residuals))
    bound,slope,offsets,residuals=min(candidates,key=lambda c:c[0])
    e1,e2,e3,ec=residuals
    # phi = slope*(A+B+C) + c1-c2+c3+d + e1[A]-e2[B]+e3[C] mod p.
    lo12,hi12=cyclic_extrema(e1,-e2)
    lo,_=cyclic_extrema(lo12,e3)
    _,hi=cyclic_extrema(hi12,e3)
    gap=(offsets[3]-(offsets[0]-offsets[1]+offsets[2]+float(memory.offset)))%period
    lower=gap+ec-hi
    upper=gap+ec-lo
    order=np.argsort(clocks%period)
    predecessors=np.empty(17,dtype=int)
    predecessors[order]=np.roll(order,1)
    spacing=(clocks-clocks[predecessors])%period
    # Positive slack puts every phase strictly inside its target clock's cell.
    slack=np.minimum(lower,spacing-upper)
    certified=bool((slack>1e-9).all())
    # Independent exhaustive contract checks the dynamic-programming extrema.
    brute=[[] for _ in range(17)]
    errors=0
    for aa in range(17):
        for bb in range(17):
            for cc in range(17):
                y=(aa+bb+cc)%17
                brute[y].append(e1[aa]-e2[bb]+e3[cc])
                pred,_=memory.serial(np.array([aa,17+bb,34+cc]))
                errors+=pred != y
    assert np.allclose(lo,[min(v) for v in brute],atol=1e-10,rtol=0)
    assert np.allclose(hi,[max(v) for v in brute],atol=1e-10,rtol=0)
    if certified:assert errors == 0
    result={"status":"completed","checkpoint":str(checkpoint),
            "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "period":period,"integer_slope":slope,"phase_intercepts":offsets,
            "phase_residuals":[e.tolist() for e in residuals],"target_clock_offset":gap,
            "sum_max_abs_residuals":bound,"simple_isometry_certificate":bool(bound<min(gap,1-gap)),
            "minplus_certificate":certified,"minimum_clock_cell_slack":float(slack.min()),
            "slack_by_class":slack.tolist(),"target_wait_lower":lower.tolist(),
            "target_wait_upper":upper.tolist(),"predecessor_spacing":spacing.tolist(),
            "exhaustive_errors":int(errors),"exhaustive_n":17**3,
            "scope":"Post-fit analytic certificate for this finite mod-17 task and guarded readout; not optimizer convergence or arbitrary modulus generalization"}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)


if __name__ == "__main__":main()
