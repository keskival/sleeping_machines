"""Frozen-parent continuation with local fine-source and temporal-phase credit.

One uniquely tagged guarded queue job. The source embedding is initialized to
zero, the old optimizer is preserved, and the old packet augmentation is kept.
Completed-utterance cross entropy; no prefix labels or official test access.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.nn import functional as F
from e139_fine_packet_model import augment_marked, load_marked, marked_batch
from e140_phase_packet_model import PhasePacketModel, phase_optimizer


@torch.no_grad()
def evaluate(model, items, bs):
    model.eval()
    correct, nll, packets, raw_events = 0, 0., 0, 0
    predictions, labels = [], []
    for start in range(0, len(items), bs):
        x = marked_batch(items[start:start+bs])
        scores, _, stats, _ = model(x)
        loss = F.cross_entropy(scores, x[4], reduction="sum")
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite evaluation loss")
        pred = scores.argmax(-1)
        correct += int((pred == x[4]).sum())
        nll += float(loss)
        packets += stats["packets"]
        raw_events += stats["source_events"]
        predictions.extend(pred.tolist())
        labels.extend(x[4].tolist())
    return {"accuracy": correct/len(items), "correct": correct, "n": len(items),
        "nll": nll/len(items), "predictions": predictions, "labels": labels,
        "input_packets": packets, "source_events": raw_events}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--checkpoint", default="experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt")
    p.add_argument("--limit", type=int, default=6144)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--core-lr", type=float, default=.00007)
    p.add_argument("--fine-lr", type=float, default=.0003)
    p.add_argument("--phase-lr", type=float, default=.003)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--checkpoint-every", type=int, default=64)
    p.add_argument("--resume-progress", action="store_true")
    a = p.parse_args()
    if Path(a.tag).name != a.tag or min(a.limit, a.epochs, a.bs, a.checkpoint_every) < 1:
        raise ValueError("Invalid run settings")
    out = Path("experiments/results/e141")/(a.tag+".json")
    out.parent.mkdir(exist_ok=True)
    progress_path = out.with_suffix(".progress.pt")
    if out.exists() and not a.resume_progress:
        raise FileExistsError(out)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    parent = torch.load(a.checkpoint, weights_only=False, map_location="cpu")
    model = PhasePacketModel(parent)
    for name,param in model.core.named_parameters():
        param.requires_grad_(name.endswith("memory_phase"))
    opt = phase_optimizer(model, parent, a.core_lr, a.fine_lr, a.phase_lr)
    fit = load_marked(a.limit, "fit_spk", a.seed)
    held = load_marked(512, "val_spk", a.seed+1)
    if len(fit) != a.limit or len(held) != 512:
        raise ValueError("Requested fitting/development size not available")
    if set(r[8] for r in fit) & set(r[8] for r in held):
        raise ValueError("Overlapping absolute fitting/development IDs")
    order_rng = np.random.default_rng(a.seed+2)
    order_rng.bit_generator.state = parent.get("numpy_rng", parent.get("order_rng"))
    aug_rng = np.random.default_rng(a.seed+122)
    if "augmentation_rng" in parent:
        aug_rng.bit_generator.state = parent["augmentation_rng"]
    sources = [Path(__file__), Path("experiments/e139_fine_packet_model.py"),
        Path("experiments/e140_phase_packet_model.py"), Path("sleeping_machines/rotating_memory.py"),
        Path("experiments/e117_serial_event_shd.py"), Path("experiments/e51_shd_world.py"),
        Path("sleeping_machines/shared_event.py"), Path("sleeping_machines/event_memory.py")]
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    started = time.perf_counter()
    def eval_splits():
        return {"fit": evaluate(model, fit, a.bs),
            "dev_original": evaluate(model, held[:256], a.bs),
            "dev_additional": evaluate(model, held[256:], a.bs)}
    if a.resume_progress:
        progress = torch.load(progress_path, weights_only=False, map_location="cpu")
        result = progress["result"]
        if result["status"] != "running" or result["source_sha256"] != source_hashes:
            raise ValueError("Resume requires identical sources and unfinished result")
        for key, value in vars(a).items():
            if key != "resume_progress" and result["args"].get(key) != value:
                raise ValueError(f"Changed resume setting: {key}")
        if result["checkpoint_sha256"] != hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest():
            raise ValueError("Changed parent checkpoint")
        model.load_state_dict(progress["state_dict"])
        opt.load_state_dict(progress["optimizer"])
        order_rng.bit_generator.state = progress["order_rng"]
        aug_rng.bit_generator.state = progress["augmentation_rng"]
        torch.set_rng_state(progress["torch_rng"])
    else:
        progress = None
        result = {"status": "running", "args": vars(a), "curve": [], "initial": eval_splits(),
            "checkpoint_sha256": hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
            "source_sha256": source_hashes, "fit_ids": [r[4] for r in fit],
            "dev_original_ids": [r[4] for r in held[:256]],
            "dev_additional_ids": [r[4] for r in held[256:]],
            "fit_absolute_ids": [r[8] for r in fit], "dev_absolute_ids": [r[8] for r in held],
            "protocol": "SHD train only; existing speaker3/6 development; clean complete-utterance evaluation; no official test",
            "augmentation": "Same legacy band shift/cropping and log-time stretch; fine slot stays relative to shifted band; raw source time scales with closure time",
            "source_representation": "700 distinct source addresses in a 40x18 embedding; original source times; 32-component summed marks; original sparse packet schedule",
            "frozen_parent_parameters": sum(p.numel() for p in model.core.parameters() if not p.requires_grad),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "credit_intervention": "Old core frozen exactly; only fine source and temporal phase trained. Clip includes only their nonzero gradients; old Adam moments preserved unused",
            "fine_parameters": model.fine.weight.numel(),
            "phase_parameters": sum(l.memory_phase.numel() for l in model.core.layers),
            "temporal_memory": "R(phase * lag/tau) damped receiver vectors; zero phase exactly nests parent normalized mean; same actual winning packets and delays",
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "normalization": "Unchanged parent fit-only calibration",
            "hardware": {"platform": platform.platform(), "torch": torch.__version__, "threads": 1},
            "energy_joules": None}
    elapsed_before = progress["elapsed_s"] if progress else 0.
    def persist():
        result.update(wall_s=elapsed_before+time.perf_counter()-started,
            max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2)+"\n")
        tmp.replace(out)
    persist()
    print(json.dumps({"initial": {k: v["accuracy"] for k, v in result["initial"].items()}}), flush=True)
    for epoch in range(progress["epoch"] if progress else 1, a.epochs+1):
        continuing = progress is not None and progress["epoch"] == epoch
        order = progress["order"] if continuing else order_rng.permutation(len(fit))
        accum = progress["accumulators"] if continuing else dict(
            nll=0., steps=0, fine_gradient=0., core_gradient=0., clipped=0,
            packets=0, source_events=0, source_payload_multiplies=0,
            source_payload_additions=0, source_packet_divisions=0,
            source_time_exponentials=0, values=0, phase_gradient=[0.]*8,
            temporal_rotation_multiplies=0, temporal_rotation_additions=0,
            temporal_angle_multiplies=0, temporal_sines=0, temporal_cosines=0,
            temporal_frequency_divisions=0)
        first = progress["next_start"] if continuing else 0
        model.train()
        for start in range(first, len(fit), a.bs):
            rows = [augment_marked(fit[j], aug_rng) for j in order[start:start+a.bs]]
            x = marked_batch(rows)
            opt.zero_grad(set_to_none=True)
            scores, _, stats, _ = model(x)
            loss = F.cross_entropy(scores, x[4])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            accum["fine_gradient"] += float(model.fine.weight.grad.norm())
            for j,layer in enumerate(model.core.layers):
                accum["phase_gradient"][j] += float(layer.memory_phase.grad.norm())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            accum["core_gradient"] += float(norm)
            accum["clipped"] += int(norm > 1.)
            opt.step()
            accum["nll"] += float(loss.detach())*len(rows)
            accum["steps"] += 1
            accum["packets"] += stats["packets"]
            accum["values"] += sum(r["value_evaluations"] for r in stats["layers"])
            for key in ("source_events", "source_payload_multiplies", "source_payload_additions",
                        "source_packet_divisions", "source_time_exponentials",
                        "temporal_rotation_multiplies", "temporal_rotation_additions",
                        "temporal_angle_multiplies", "temporal_sines", "temporal_cosines",
                        "temporal_frequency_divisions"):
                accum[key] += stats[key]
            next_start = min(start+a.bs, len(fit))
            if accum["steps"] % a.checkpoint_every == 0 or next_start == len(fit):
                result["progress"] = {"epoch": epoch, "completed_examples": next_start,
                    "steps": accum["steps"], "online_nll": accum["nll"]/next_start}
                persist()
                state = {"state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                    "result": result, "epoch": epoch, "next_start": next_start,
                    "order": order, "order_rng": order_rng.bit_generator.state,
                    "augmentation_rng": aug_rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "elapsed_s": result["wall_s"],
                    "accumulators": accum}
                temporary = progress_path.with_suffix(".pt.tmp")
                torch.save(state, temporary)
                temporary.replace(progress_path)
                print(json.dumps(result["progress"]), flush=True)
        for name,value in parent["state_dict"].items():
            if not torch.equal(model.core.state_dict()[name],value):
                raise ValueError(f"Frozen parent changed: {name}")
        row = {"parent_state_unchanged": True, "epoch": epoch, "online_nll": accum["nll"]/len(fit),
            "phase_parameter_norm_per_layer": [float(l.memory_phase.detach().norm()) for l in model.core.layers],
            "training": accum, "fine_parameter_norm": float(model.fine.weight.detach().norm()),
            **eval_splits()}
        result["curve"].append(row)
        progress = None
        checkpoint = {"args": vars(a), "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(), "source_sha256": source_hashes,
            "order_rng": order_rng.bit_generator.state, "augmentation_rng": aug_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(), "epoch": epoch, "curve": result["curve"],
            "parent_checkpoint": a.checkpoint, "parent_sha256": result["checkpoint_sha256"]}
        tmp = out.with_suffix(".pt.tmp")
        torch.save(checkpoint, tmp)
        tmp.replace(out.with_suffix(".pt"))
        persist()
        print(json.dumps({"epoch": epoch, "fit": row["fit"]["accuracy"],
            "original": row["dev_original"]["accuracy"],
            "additional": row["dev_additional"]["accuracy"], "wall_s": result["wall_s"]}), flush=True)
    result.update(status="completed", final=result["curve"][-1])
    persist()


if __name__ == "__main__":
    main()
