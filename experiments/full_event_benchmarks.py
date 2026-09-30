"""Full declared-data benchmarks for learned event models, without count experts.

Run only via a one-job guarded queue. Selection is on development; official
tests are scored once after the fixed epoch budget. Checkpoints are resumable.
"""
import argparse
import copy
import csv
import gzip
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
from e120_shared_tasks import Example, Task, prefix, market, temporal
from e120_shared_bench import calibrate, inputs, loss_for
from e120_dvs_adapter import recording
from e139_fine_packet_model import marked_packets, augment_marked
from e143_event_state_shd import event_batch
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.event_state import CoalescedEventStateEncoder


def mnist_rows(split):
    image = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    label = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    with gzip.open(ROOT / "data" / image, "rb") as f:
        f.read(16)
        pixels = np.frombuffer(f.read(), np.uint8).reshape(-1, 28, 28).astype(np.float32) / 255
    with gzip.open(ROOT / "data" / label, "rb") as f:
        f.read(8)
        labels = np.frombuffer(f.read(), np.uint8)
    pooled = pixels.reshape(-1, 14, 2, 14, 2).mean((2, 4)).reshape(-1, 196)
    rows = []
    for i, (x, y) in enumerate(zip(pooled, labels)):
        channels = np.flatnonzero(x > .1)
        times = (1 - x[channels]) * .8
        order = np.argsort(times, kind="stable")
        rows.append(Example(prefix(channels[order], times[order]), int(y), f"{split}:{i}"))
    return rows


def dvs_rows(split, users=None):
    root = ROOT / "data/dvsgesture/DvsGesture"
    rows = []
    for name in (root / f"trials_to_{split}.txt").read_text().splitlines():
        if name.endswith(".aedat") and (users is None or int(name[4:6]) in users):
            part, _ = recording(root / name, observation_seconds=None)
            rows.extend(part)
    return rows


class SHDRows:
    """Read each requested utterance; never retain a full raw-event corpus."""
    def __init__(self, split, held=None):
        self.file = h5py.File(ROOT / f"data/shd/shd_{split}.h5", "r")
        self.times = self.file["spikes"]["times"]
        self.units = self.file["spikes"]["units"]
        self.labels = np.asarray(self.file["labels"])
        if held is None:
            self.indices = np.arange(len(self.labels))
        else:
            speakers = np.asarray(self.file["extra"]["speaker"])
            self.indices = np.flatnonzero(np.isin(speakers, (3, 6)) == held)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        i = int(self.indices[index])
        times = np.asarray(self.times[i], np.float64)
        units = np.asarray(self.units[i], np.int64)
        order = np.argsort(times, kind="stable")
        b, t, c, assignment, fine, raw = marked_packets(times[order], units[order])
        return b, t, c, int(self.labels[i]), i, assignment, fine, raw, i


def shd_rows(split, held=None):
    return SHDRows(split, held)


def encoded_hash(rows, speech=False):
    digest = hashlib.sha256()
    for row in rows:
        arrays = (row[0], row[1], row[2], row[5], row[6], row[7]) if speech else (
            row.prefix.channels, row.prefix.times, row.prefix.counts)
        for array in arrays:
            array = np.asarray(array)
            digest.update(str((array.dtype.str, array.shape)).encode())
            digest.update(array.tobytes())
        digest.update(str(row[3] if speech else row.label).encode())
        if not speech and row.exposure is not None:
            digest.update(np.asarray(row.exposure).tobytes())
            digest.update(str(row.bucket).encode())
    return digest.hexdigest()


def load_task(name, seed):
    if name == "shd":
        return shd_rows("train", False), shd_rows("train", True), dict(
            dataset="SHD", fit="all 6987 train-file utterances outside speakers 3/6",
            dev="all 1169 train-file utterances from speakers 3/6",
            encoding="injective 700-channel source addresses, original time, nonempty 10ms global closures",
            inheritance="none; from-scratch six-block learned event-state encoder")
    if name == "mnist":
        rows = mnist_rows("train")
        return rows[:50000], rows[50000:], dict(dataset="MNIST", fit=[0, 50000], dev=[50000, 60000],
            encoding="declared 2x2 average pooling, intensity threshold .1, latency .8*(1-intensity)")
    if name == "dvs":
        return dvs_rows("train", range(1, 20)), dvs_rows("train", range(20, 24)), dict(
            dataset="DVS128 Gesture", fit="all 984 gestures from users 1..19",
            dev="all 192 gestures from users 20..23", observation="complete labelled gesture",
            encoding="declared 4x4 cells, polarity and count packets at 50ms closures")
    task = market(None, None, seed, evidence_free=True, raw_limit=None) if name == "market" else temporal(32768, 8192, seed)
    return task.fit, task.dev, task.protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("shd", "mnist", "dvs", "market", "temporal"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--contract", action="store_true", help="Eight fitting/four dev examples; no official test")
    args = parser.parse_args()
    out = ROOT / "experiments/results/full_event" / (args.tag + ".json")
    running = out.with_suffix(".running.json")
    checkpoint = out.with_suffix(".progress.pt")
    if out.exists() and json.loads(out.read_text()).get("status") == "completed":
        raise ValueError("Completed results must never be rerun or overwritten")
    if Path(args.tag).name != args.tag or args.epochs < 1 or ((out.exists() or running.exists()) and not args.resume):
        raise ValueError("Unique tag and positive epoch budget required")
    out.parent.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    fit, dev, protocol = load_task(args.task, args.seed)
    if args.contract:
        fit, dev = fit[:8], dev[:4]
        protocol = {**protocol, "contract_only": True, "official_test_read": False}
    if args.task != "shd" and any(r.evidence is not None for r in fit + dev):
        raise ValueError("Statistical experts are forbidden")
    bs = 4 if args.task in ("shd", "dvs") else 16
    if args.task == "shd":
        model = CoalescedEventStateEncoder(sources=720, width=128, modes=64, depth=6)
    else:
        config = {"mnist": dict(bands=196, classes=10, groups=4),
                  "dvs": dict(bands=33, classes=11, groups=1),
                  "market": dict(bands=4, classes=24, groups=1, readout="last"),
                  "temporal": dict(bands=8, classes=2, groups=1)}[args.task]
        if args.task == "temporal":
            config = {k: v for k, v in temporal(1, 1, args.seed).config.items() if k != "evidence_count"}
        model = SharedEventModel(**config, dim=32, depth=8, memory_backend="linear", evidence_count=0)
        calibrate(model, fit, bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001 if args.task == "shd" else .003)
    rng = np.random.default_rng(args.seed + 10)
    augment_rng = np.random.default_rng(args.seed + 122)

    def predict(rows):
        if args.task == "shd":
            return model(*event_batch(rows, .01), len(rows))[0], torch.tensor([r[3] for r in rows])
        return model(**inputs(rows))[0], None

    @torch.no_grad()
    def evaluate(rows):
        model.eval()
        total, correct = 0., 0
        for start in range(0, len(rows), bs):
            part = rows[start:start + bs]
            scores, y = predict(part)
            total += float(F.cross_entropy(scores, y, reduction="sum") if y is not None else loss_for(scores, part, "sum"))
            if args.task != "market":
                labels = y if y is not None else torch.tensor([r.label for r in part])
                correct += int((scores.argmax(-1) == labels).sum())
        result = dict(n=len(rows), nll=total / len(rows))
        if args.task != "market":
            result.update(correct=correct, accuracy=correct / len(rows))
        return result

    paths = [Path(__file__)] + [ROOT / p for p in (
        "sleeping_machines/shared_event.py", "sleeping_machines/event_state.py",
        "sleeping_machines/event_memory.py", "sleeping_machines/rotating_memory.py",
        "sleeping_machines/event_query.py", "sleeping_machines/objectives.py",
        "sleeping_machines/readout_calibration.py", "experiments/e120_shared_tasks.py",
        "experiments/e120_shared_bench.py", "experiments/e120_dvs_adapter.py",
        "experiments/e139_fine_packet_model.py", "experiments/e143_event_state_shd.py")]
    result = dict(status="running", args=vars(args), protocol=protocol, statistical_experts=[],
        parameters=sum(p.numel() for p in model.parameters()), fitting_examples=len(fit), development_examples=len(dev),
        encoded_data_sha256=dict(fit=encoded_hash(fit, args.task == "shd"), dev=encoded_hash(dev, args.task == "shd")),
        curve=[], hardware=dict(platform=platform.platform(), torch=torch.__version__, device="cpu", threads=1),
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope=("Small execution contract, not benchmark evidence" if args.contract else
               "Full declared split, one seed, proper learned event model; preprocessing and capacity are declared, not a frontier claim"),
        test_policy="Fixed epoch budget; minimum development NLL checkpoint; official test once after selection where available")
    epoch, next_start, order, accumulated, best_loss, best_state = 1, 0, None, 0., float("inf"), None
    previous_wall = 0.
    if checkpoint.exists() and args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in ("task", "tag", "epochs", "seed", "contract"):
            if saved["result"]["args"][key] != vars(args)[key]:
                raise ValueError("Changed resumed settings")
        if saved["result"]["source_sha256"] != result["source_sha256"]:
            raise ValueError("Changed resumed implementation")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        result = saved["result"]
        rng.bit_generator.state = saved["rng"]
        augment_rng.bit_generator.state = saved["augment_rng"]
        torch.set_rng_state(saved["torch_rng"])
        epoch, next_start, order, accumulated, best_loss, best_state = (saved[k] for k in
            ("epoch", "next_start", "order", "accumulated", "best_loss", "best_state"))
        previous_wall = result["wall_s"]
    elif out.exists() or running.exists():
        raise ValueError("Existing output lacks a resumable checkpoint")

    def persist(save=False):
        result.update(wall_s=previous_wall + time.perf_counter() - started,
                      max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        target = out if result["status"] == "completed" else running
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(target)
        if result["status"] == "completed":
            running.unlink(missing_ok=True)
        if save:
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), result=result,
                epoch=epoch, next_start=next_start, order=order, accumulated=accumulated,
                best_loss=best_loss, best_state=best_state, rng=rng.bit_generator.state,
                augment_rng=augment_rng.bit_generator.state, torch_rng=torch.get_rng_state()), temporary)
            temporary.replace(checkpoint)
    persist()
    print(json.dumps(dict(task=args.task, fit=len(fit), dev=len(dev), parameters=result["parameters"])), flush=True)
    while epoch <= args.epochs:
        if order is None:
            order = rng.permutation(len(fit))
            next_start, accumulated = 0, 0.
        model.train()
        for start in range(next_start, len(fit), bs):
            rows = [fit[i] for i in order[start:start + bs]]
            if args.task == "shd":
                rows = [augment_marked(r, augment_rng) for r in rows]
            optimizer.zero_grad(set_to_none=True)
            scores, y = predict(rows)
            loss = F.cross_entropy(scores, y) if y is not None else loss_for(scores, rows)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            accumulated += float(loss.detach()) * len(rows)
            next_start = start + len(rows)
            if (start // bs + 1) % 128 == 0:
                result["progress"] = dict(epoch=epoch, examples=next_start, online_nll=accumulated / next_start)
                persist(True)
                print(json.dumps(result["progress"]), flush=True)
        score = evaluate(dev)
        result["curve"].append(dict(epoch=epoch, fitting_online_nll=accumulated / len(fit), dev=score))
        if score["nll"] < best_loss:
            best_loss, best_state = score["nll"], copy.deepcopy(model.state_dict())
            result["selected_epoch"] = epoch
        print(json.dumps(result["curve"][-1]), flush=True)
        epoch, order = epoch + 1, None
        persist(True)
    model.load_state_dict(best_state)
    result["final"] = dict(dev=evaluate(dev))
    if args.task in ("mnist", "dvs", "shd") and not args.contract:
        test = mnist_rows("test") if args.task == "mnist" else dvs_rows("test") if args.task == "dvs" else shd_rows("test")
        result["encoded_data_sha256"]["test"] = encoded_hash(test, args.task == "shd")
        result["final"]["official_test"] = evaluate(test)
    elif args.task == "market" and not args.contract:
        future = market(1, None, args.seed, evidence_free=True, raw_limit=None,
                        dev_day="2026-08-30").dev
        result["encoded_data_sha256"]["future_day"] = encoded_hash(future)
        result["final"]["future_day_20260830"] = evaluate(future)
        result["protocol"]["final_future_day"] = "2026-08-30; frozen selected model, no reselection"
    result["status"] = "completed"
    persist(True)
    print(json.dumps(result["final"]), flush=True)


if __name__ == "__main__":
    main()
