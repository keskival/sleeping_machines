"""Frozen intervention on unsupported readout coordinates; no parameter updates."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.readout_calibration import condition_readout
from e120_shared_bench import inputs, evaluate
from e120_shared_tasks import recall


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    args = p.parse_args()
    out = Path("experiments/results/e120")/(args.tag+".json")
    if out.exists():
        raise FileExistsError(out)
    checkpoint = Path("experiments/results/e120/recall_d8_20260929.pt")
    saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
    a = saved["args"]
    torch.set_num_threads(1)
    task = recall(a["fit"], a["dev"], a["seed"])
    net = SharedEventModel(**saved["config"])
    net.load_state_dict(saved["state_dict"])
    net.eval()
    # For every training context the count coordinate was exactly the same.
    # Remove only that coordinate's contribution, leaving all learned weights.
    fixed = copy.deepcopy(net)
    fixed.whitener[-1, :] = 0
    # The replacement fit-only rule must reproduce this intervention and
    # leave variable-count calibration identical to the old rule.
    probe = SharedEventModel(**saved["config"])
    toy = torch.randn(100, probe.dim+1, dtype=torch.float64)
    toy[:, -1] = .3
    audit = condition_readout(probe, toy)
    assert audit["constant_count_projected"] and torch.count_nonzero(probe.whitener[-1]) == 0
    toy[:, -1] = torch.arange(100, dtype=torch.float64)/100
    condition_readout(probe, toy)
    other = copy.deepcopy(probe)
    condition_readout(other, toy, project_constant_count=False)
    assert torch.equal(probe.whitener, other.whitener)
    result = {"status": "completed", "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "intervention": "zero only the input row of the whitener for log event count; no training",
              "count_scale": float(net.scale[-1]), "count_center": float(net.center[-1]), "splits": {}}
    result["calibration_contracts"] = {"constant_count_projected": True,
                                       "variable_count_calibration_unchanged": True}
    for name, rows in {"fit": task.fit, "dev": task.dev, **task.extra}.items():
        packed = inputs(rows)
        scores, summary, _, _ = net(**packed)
        contribution = ((summary[:, -1]-net.center[-1])/net.scale[-1])[:, None]*net.whitener[-1][None]
        result["splits"][name] = {"before": evaluate(net, rows, 16), "count_projected": evaluate(fixed, rows, 16),
            "count_feature_normalized_max": float(((summary[:, -1]-net.center[-1])/net.scale[-1]).abs().max()),
            "count_head_score_max": float((contribution@net.head.weight.T).abs().max()),
            "count_gate_score_max": float((contribution@net.evidence_gate.weight.T).abs().max())}
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({name: {"before": v["before"]["accuracy"], "projected": v["count_projected"]["accuracy"],
                            "normalized_count": v["count_feature_normalized_max"],
                            "count_head_score_max": v["count_head_score_max"]}
                      for name, v in result["splits"].items()}))


if __name__ == "__main__":
    main()
