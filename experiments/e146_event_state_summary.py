"""Completed-result/provenance summary of the temporal residual SHD experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument("--tag",required=True);a=p.parse_args()
    out=Path("experiments/results/e146")/(a.tag+".json");out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError("Unique output required")
    paths={"training":"experiments/results/e143/d8_parent_d6_state_residual_n6144_s6_e3_20260930.json",
        "ablation":"experiments/results/e145/state_residual_ablation_20260930.json",
        "coalescing":"experiments/results/e142/event_state_contract_v2_20260930.json",
        "pooling":"experiments/results/e144/affine_pooling_contract_20260930.json",
        "disjoint_audit":"experiments/results/e147/disjoint_speaker_audit_20260930.json"}
    data={k:json.loads(Path(v).read_text()) for k,v in paths.items()}
    for r in data.values():
        assert r["status"]=="completed"
        for source,digest in r["source_sha256"].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest,source
    run=data["training"];selected=max(run["curve"],key=lambda r:r["dev"]["correct"])
    assert data["ablation"]["selected_epoch"]==selected["epoch"]
    assert set(run["fit_absolute_ids"]).isdisjoint(run["dev_absolute_ids"])
    audit=data["disjoint_audit"]
    assert set(audit["absolute_ids"]).isdisjoint(run["fit_absolute_ids"]+run["dev_absolute_ids"])
    pred=selected["dev"]["predictions"];parent=run["initial"]["dev"]["predictions"]
    labels=selected["dev"]["labels"];n=len(labels)
    assert labels==run["initial"]["dev"]["labels"]
    new_only=sum(p==y and q!=y for p,q,y in zip(pred,parent,labels))
    parent_only=sum(p!=y and q==y for p,q,y in zip(pred,parent,labels))
    delta=(new_only-parent_only)/n
    assert selected["dev"]["correct"]-run["initial"]["dev"]["correct"]==new_only-parent_only
    curve=[]
    for row in run["curve"]:
        assert row["parent_state_unchanged"]
        training=row["training"]
        curve.append(dict(epoch=row["epoch"],fit_correct=row["fit"]["correct"],fit_n=row["fit"]["n"],
            fit_nll=row["fit"]["nll"],held_correct=row["dev"]["correct"],held_n=row["dev"]["n"],
            held_nll=row["dev"]["nll"],learning_rate=row["lr"],
            old_packets=training["old_packets"],new_packets=training["new_packets"],
            source_events=training["source_events"],source_table_projection_macs=training["source_table_projection_macs"],
            new_event_projection_macs=training["event_projection_macs"],new_gate_macs=training["gate_macs"],
            new_state_compositions=training["state_compositions"],new_clock_candidates=training["clock_candidates"],
            training_wall_s=training["training_wall_s"],layer_gradient_norm_sums=training["layer_gradient_norm"]))
    guards={}
    for job in ("e143_event_state_residual_20260930","e144_affine_pooling_contract_20260930","e145_event_state_ablation_20260930","e147_disjoint_speaker_audit_20260930"):
        log=Path("experiments/queue")/("runner_"+job+".out")
        content=log.read_text()
        assert f"done {job} (exit 0)" in content
        available=[int(x) for x in re.findall(r"MemAvailable=(\d+)MB",content)]
        guards[job]=dict(log=str(log),memory_floor_mib=8192,rss_cap_kib=2600000,
            address_space_cap_kib=4200000 if job.startswith("e143") else 3600000,
            timeout_s=3600 if job.startswith("e143") else 600,
            minimum_sampled_available_mib=min(available) if available else None)
    result=dict(status="completed",selected_epoch=selected["epoch"],parent_correct=run["initial"]["dev"]["correct"],
        selected_correct=selected["dev"]["correct"],final_correct=run["final"]["dev"]["correct"],n=n,
        selected_gain_percentage_points=100*delta,new_only_correct=new_only,parent_only_correct=parent_only,
        selection="Best of three private-development epochs, one seed, two held training speakers; no official-test or global SOTA claim",
        curve=curve,old_parameters=run["old_parameters"],new_parameters=run["new_parameters"],
        total_parameters=run["old_parameters"]+run["new_parameters"],wall_s=run["wall_s"],max_rss_kb=run["max_rss_kb"],
        ablation={k:dict(correct=v["held"]["correct"],nll=v["held"]["nll"],paired_vs_trained=v["paired_vs_trained"])
            for k,v in data["ablation"]["rows"].items()},
        reset_scope=data["ablation"]["scope"],resource_guards=guards,work_scope=run["work_scope"],
        disjoint_audit=dict(n=audit["n"],parent_correct=audit["parent"]["correct"],
            residual_correct=audit["residual"]["correct"],parent_nll=audit["parent"]["nll"],
            residual_nll=audit["residual"]["nll"],gain_percentage_points=audit["gain_percentage_points"],
            scope=audit["scope"]),
        timing_scope="E143 timer includes preparation, training and initial/epoch evaluation; per-epoch training_wall_s excludes evaluation",
        energy_joules=None,result_files=paths,
        result_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in paths.items()},
        source_sha256={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k in ("status","selected_epoch","parent_correct","selected_correct","final_correct","selected_gain_percentage_points","new_only_correct","parent_only_correct","total_parameters","wall_s","max_rss_kb")}),flush=True)


if __name__=="__main__":main()
