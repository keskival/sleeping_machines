"""One task per guarded job; one shared architecture, independently fitted weights."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.event_query import pack_queries
from sleeping_machines.objectives import categorical_nll, hazard_nll
from sleeping_machines.readout_calibration import condition_readout
from e120_shared_tasks import BUILDERS
from e120_dvs_adapter import dvs

BUILDERS = {**BUILDERS, "dvs": dvs}


def inputs(rows):
    batch = pack_queries([r.prefix for r in rows])
    if rows[0].evidence is not None:
        batch["evidence"] = torch.from_numpy(np.stack([r.evidence for r in rows])).float()
    return batch


def loss_for(scores, rows, reduction="mean"):
    y = torch.tensor([r.label for r in rows])
    if rows[0].exposure is None:
        return categorical_nll(scores, y, reduction)
    exposure = torch.from_numpy(np.stack([r.exposure for r in rows])).float()
    k = torch.tensor([r.bucket for r in rows])
    return hazard_nll(scores.reshape(len(rows), -1, 4), y, k, exposure, reduction)


@torch.no_grad()
def calibrate(model, rows, bs, project_constant_count=True):
    model.eval()
    x = torch.cat([model(**inputs(rows[i:i+bs]))[1] for i in range(0, len(rows), bs)]).double()
    return condition_readout(model, x, project_constant_count)


@torch.no_grad()
def evaluate(model, rows, bs, mode="combined"):
    model.eval()
    total, correct, packets, scans, values = 0., 0, 0, 0, 0
    started = time.perf_counter()
    predictions = []
    for i in range(0, len(rows), bs):
        part = rows[i:i+bs]
        packed = inputs(part)
        if mode == "fixed_evidence":
            scores = packed["evidence"].mean(1)
        else:
            scores, _, stats, _ = model(**packed, expert_mode=mode)
            packets += stats["packets"]
            scans += sum(x["scan_compositions"] for x in stats["layers"])
            values += sum(x["value_evaluations"] for x in stats["layers"])
        total += float(loss_for(scores, part, "sum"))
        if rows[0].exposure is None:
            pred = scores.argmax(-1).tolist()
            predictions.extend(pred)
            correct += sum(p == r.label for p, r in zip(pred, part))
    result = {"n": len(rows), "nll": total/len(rows), "wall_s": time.perf_counter()-started,
              "prefix_packets_processed": packets, "scan_compositions": scans,
              "selected_value_evaluations": values}
    if rows[0].exposure is None:
        result.update(correct=correct, accuracy=correct/len(rows), predictions=predictions)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=BUILDERS, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--fit", type=int, default=512)
    p.add_argument("--dev", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr", type=float, default=.003)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--value-backward", choices=("full", "winner"), default="full")
    p.add_argument("--legacy-count-calibration", action="store_true",
                   help="reproduce the initial screen's unsupported count scaling")
    args = p.parse_args()
    if Path(args.tag).name != args.tag or min(args.fit, args.dev, args.epochs, args.bs) < 1:
        raise ValueError("Invalid run settings")
    out = Path("experiments/results/e120")/(args.tag+".json")
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    task = BUILDERS[args.task](args.fit, args.dev, args.seed)
    config = {**task.config, "dim": args.dim, "depth": args.depth, "memory_backend": "linear",
              "value_backward": args.value_backward}
    model = SharedEventModel(**config)
    calibration = calibrate(model, task.fit, args.bs, not args.legacy_count_calibration)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.lr/10)
    rng = np.random.default_rng(args.seed+10)
    source = [Path(__file__), Path("experiments/e120_shared_tasks.py"), Path("experiments/e120_dvs_adapter.py"),
              *Path("sleeping_machines").glob("*.py")]
    result = {"status": "running", "args": vars(args), "config": config, "protocol": task.protocol,
              "parameters": sum(p.numel() for p in model.parameters()), "calibration": calibration,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
              "fit_ids": [r.identity for r in task.fit], "dev_ids": [r.identity for r in task.dev],
              "initial": {"fit": evaluate(model, task.fit, args.bs), "dev": evaluate(model, task.dev, args.bs)},
              "curve": [], "hardware": {"platform": platform.platform(), "torch": torch.__version__,
              "threads": 1, "device": "cpu"}, "energy_joules": None,
              "cost_note": "prefix replay, sorting, and local vector maps; timings are not measured energy"}
    def persist():
        temp = out.with_suffix(".json.tmp")
        temp.write_text(json.dumps(result, indent=2)+"\n")
        temp.replace(out)
    persist()
    print(json.dumps({"task": args.task, "initial": result["initial"]["dev"]["nll"]}), flush=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        total, packets, scans, values = 0., 0, 0, 0
        grad = np.zeros(args.depth)
        order = rng.permutation(len(task.fit))
        for i in range(0, len(order), args.bs):
            rows = [task.fit[j] for j in order[i:i+args.bs]]
            scores, _, stats, _ = model(**inputs(rows))
            loss = loss_for(scores, rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite likelihood")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for j, layer in enumerate(model.layers):
                grad[j] += float(layer.route.grad.norm())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(rows)
            packets += stats["packets"]
            scans += sum(x["scan_compositions"] for x in stats["layers"])
            values += sum(x["value_evaluations"] for x in stats["layers"])
        scheduler.step()
        row = {"epoch": epoch, "online_nll": total/len(task.fit),
               "fit": evaluate(model, task.fit, args.bs), "dev": evaluate(model, task.dev, args.bs),
               "router_gradient_norm_sum": grad.tolist(), "training_prefix_packets": packets,
               "training_scan_compositions": scans, "training_value_evaluations": values}
        result["curve"].append(row)
        persist()
        print(json.dumps({"task": args.task, "epoch": epoch, "fit_nll": row["fit"]["nll"],
                          "dev_nll": row["dev"]["nll"], "dev_accuracy": row["dev"].get("accuracy")}), flush=True)
    result["final"] = result["curve"][-1]
    result["extra"] = {name: evaluate(model, rows, args.bs) for name, rows in task.extra.items()}
    if model.evidence_count:
        result["ablations"] = {mode: evaluate(model, task.dev, args.bs, mode)
                               for mode in ("core", "memory", "fixed_evidence")}
        result["ablation_note"] = ("Frozen readout deletions, not retrained controls. 'memory' keeps the "
                                   "deep-feature gate; 'fixed_evidence' uses equal expert weights with no deep core.")
        result["extra_fixed_evidence"] = {name: evaluate(model, rows, args.bs, "fixed_evidence")
                                          for name, rows in task.extra.items()}
        if isinstance(task.memory, list):
            result["evidence_work"] = [m.work() for m in task.memory]
        else:
            result["evidence_work"] = {"route_candidates": task.memory.candidates,
                "memory_training_mistakes": task.memory.mistakes, "state_bytes": task.memory.weights.nbytes}
        result["evidence_work_note"] = "Memory fit and precomputed evidence included; lookup preparation excluded from eval wall_s"
    torch.save({"state_dict": model.state_dict(), "config": config, "args": vars(args),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                "evidence_memory": task.memory, "protocol": task.protocol,
                "source_sha256": result["source_sha256"]}, out.with_suffix(".pt"))
    result.update(status="completed", wall_s=time.perf_counter()-started,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    persist()
    print(json.dumps({"completed": args.task, "wall_s": result["wall_s"], "max_rss_kb": result["max_rss_kb"]}), flush=True)


if __name__ == "__main__":
    main()
