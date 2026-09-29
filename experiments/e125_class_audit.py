"""Separate fitting and new-speaker class errors in completed SHD results."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
import e51_shd_world as S


def class_scores(predictions, labels):
    predictions = np.asarray(predictions)
    confusion = np.zeros((20, 20), dtype=int)
    np.add.at(confusion, (labels, predictions), 1)
    rows = []
    for label in range(20):
        n = int(confusion[label].sum())
        errors = sorted([(int(v), int(k)) for k, v in enumerate(confusion[label])
                         if k != label and v], reverse=True)
        rows.append({"class": label, "n": n, "correct": int(confusion[label, label]),
                     "accuracy": float(confusion[label, label] / n) if n else None,
                     "largest_errors": [{"predicted_class": k, "count": v} for v, k in errors[:3]]})
    return {"classes": rows, "confusion": confusion.tolist(),
            "correct": int(np.trace(confusion)), "n": int(confusion.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    out = Path("experiments/results/e125") / (args.tag + ".json")
    if out.exists():
        raise FileExistsError(out)
    paths = [Path("experiments/results/e122") / name for name in (
        "d8_n4096_invariance_continue_s6_e2.json", "d8_n4096_pool_control_s6.json",
        "d8_n4096_pool_weighted_s6.json")]
    with h5py.File(Path(S.ROOT) / "shd_train.h5", "r") as data:
        held = np.isin(np.array(data["extra"]["speaker"]), S.VAL_SPEAKERS)
        labels = np.array(data["labels"])
        speakers = np.array(data["extra"]["speaker"])
    rows = {}
    for name, path in zip(("starting_checkpoint", "mean_continuation", "weighted_continuation"), paths):
        result = json.loads(path.read_text())
        assert result["status"] == "completed"
        entry = {}
        for split in ("fit", "held"):
            parts = ["fit"] if split == "fit" else ["dev_original", "dev_additional"]
            source = ~held if split == "fit" else held
            ids = np.concatenate([np.asarray(result[s + "_ids"], dtype=int) for s in parts])
            y, spk = labels[source][ids], speakers[source][ids]
            pred = np.array(sum((result["final"][s]["predictions"] for s in parts), []))
            stats = class_scores(pred, y)
            assert stats["correct"] == sum(result["final"][s]["correct"] for s in parts)
            stats["by_speaker"] = {str(k): class_scores(pred[spk == k], y[spk == k])
                                   for k in sorted(set(spk.tolist()))}
            entry[split] = stats
        rows[name] = entry
    record = {"status": "completed", "rows": rows,
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), *paths]},
              "scope": "No retraining; existing fit and 512 held-out training-file utterances; official test untouched; descriptive class/speaker errors"}
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
