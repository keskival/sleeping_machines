"""Completed same-parent SHD results and preserved host resource evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    out = Path("experiments/results/e140")/(a.tag+".json")
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError("Unique output required")
    paths = {
        "warm": Path("experiments/results/e122/d8_n6144_best_warm_s6_e1_20260930.json"),
        "fine": Path("experiments/results/e139/d8_fine_source_n6144_warm_s6_e1_20260930.json"),
        "phase": Path("experiments/results/e140/d8_phase_source_n6144_warm_s6_e1_20260930.json")}
    records = {name: json.loads(path.read_text()) for name,path in paths.items()}
    if any(r["status"] != "completed" for r in records.values()):
        raise ValueError("Only completed result files allowed")
    reference = records["fine"]
    def concatenate(record, endpoint, key):
        return np.concatenate([record[endpoint][split][key] for split in
            ("dev_original", "dev_additional")])
    labels = concatenate(reference,"final","labels")
    initial = concatenate(reference,"initial","predictions")
    for row in records.values():
        for key in ("fit_ids", "dev_original_ids", "dev_additional_ids", "checkpoint_sha256"):
            assert row[key] == reference[key], key
        assert np.array_equal(concatenate(row,"initial","predictions"), initial)
        for key in ("limit", "epochs", "bs", "seed", "checkpoint"):
            assert row["args"][key] == reference["args"][key], key
    rows = {}
    hits = {}
    for name,row in records.items():
        final = row["final"]
        pred = concatenate(row,"final","predictions")
        hit = pred == labels
        assert int(hit.sum()) == sum(final[s]["correct"] for s in ("dev_original","dev_additional"))
        hits[name] = hit
        rows[name] = {"held_correct":int(hit.sum()),"held_n":len(hit),"held_accuracy":float(hit.mean()),
            "held_nll":sum(final[s]["nll"]*final[s]["n"] for s in
                ("dev_original","dev_additional"))/len(hit),
            "fit_accuracy":final["fit"]["accuracy"],"fit_nll":final["fit"]["nll"],
            "wall_s":row["wall_s"],"max_rss_kb":row["max_rss_kb"],
            "beats_parent_record":int(hit.sum()) > int((initial == labels).sum())}
    paired = {}
    for left,right in (("fine","warm"),("phase","fine"),("phase","warm")):
        delta = hits[left].astype(float)-hits[right]
        paired[f"{left}_vs_{right}"]={"only_left_correct":int((hits[left]&~hits[right]).sum()),
            "only_right_correct":int((hits[right]&~hits[left]).sum()),
            "accuracy_difference":float(delta.mean()),
            "descriptive_paired_se":float(delta.std(ddof=1)/np.sqrt(len(delta))),
            "scope":"Fixed examples and seed; does not capture speaker or training-seed population uncertainty"}
    jobs={"warm":"e138_best_scale_20260930","fine":"e139_fine_source_20260930",
          "phase":"e140_phase_source_20260930"}
    guards={}
    for name,job in jobs.items():
        path=Path("experiments/queue")/f"runner_{job}.out"
        text=path.read_text()
        samples=re.findall(r"process group RSS=(\d+)KB, MemAvailable=(\d+)MB",text)
        assert f"done {job} (exit 0)" in text and "STOP " not in text
        minimum=min(int(avail) for _,avail in samples)
        assert minimum >= 8192
        guards[name]={"runner_log":str(path),"sampled_min_mem_available_mib":minimum,
            "sampled_max_group_rss_kib":max(int(rss) for rss,_ in samples),
            "mem_cap_kb":3600000,"group_rss_cap_kb":2600000,
            "mem_available_floor_mib":8192,"timeout_s":1800,
            "lock":"/tmp/experiments-runner.lock","exit_code":0}
    result={"status":"completed","parent_correct":int((initial==labels).sum()),
        "parent_accuracy":float((initial==labels).mean()),"rows":rows,"paired":paired,
        "resource_guards":guards,
        "input_sha256":{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths.values()},
        "scope":"One-pass same-parent source/phase interventions on private held training speakers; no official-test parity or accuracy record improvement"}
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__ == "__main__":main()
