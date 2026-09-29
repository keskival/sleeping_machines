"""Paired held-out accounting for matched event-readout continuations."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from e122_summary import paired
import e51_shd_world as S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    out = Path("experiments/results/e125") / (args.tag + ".json")
    if out.exists():
        raise FileExistsError(out)
    paths = [Path("experiments/results/e122") / f"d8_n4096_pool_{arm}_s6.json"
             for arm in ("control", "weighted")]
    mean, weighted = [json.loads(path.read_text()) for path in paths]
    assert mean["status"] == weighted["status"] == "completed"
    for key in ("checkpoint_sha256", "fit_ids", "dev_original_ids", "dev_additional_ids", "initial"):
        assert mean[key] == weighted[key], key
    for key in ("limit", "epochs", "bs", "lr", "seed", "augment"):
        assert mean["args"][key] == weighted["args"][key], key
    with h5py.File(Path(S.ROOT) / "shd_train.h5", "r") as data:
        selected = np.isin(np.array(data["extra"]["speaker"]), S.VAL_SPEAKERS)
        labels = np.array(data["labels"])[selected]
        speakers = np.array(data["extra"]["speaker"])[selected]
    comparisons = {}
    for split in ("dev_original", "dev_additional", "pooled"):
        parts = [split] if split != "pooled" else ["dev_original", "dev_additional"]
        ids = np.concatenate([np.array(mean[s + "_ids"]) for s in parts])
        y = labels[ids]
        predictions = lambda model, endpoint: np.array(sum(
            (model[endpoint][s]["predictions"] for s in parts), []))
        initial = predictions(mean, "initial")
        a, b = predictions(mean, "final"), predictions(weighted, "final")
        entry = {"starting_vs_mean": paired(initial, a, y),
                 "starting_vs_weighted": paired(initial, b, y),
                 "mean_vs_weighted": paired(a, b, y)}
        if split == "pooled":
            entry["by_speaker"] = {str(s): paired(a[speakers[ids] == s], b[speakers[ids] == s],
                                                y[speakers[ids] == s]) for s in S.VAL_SPEAKERS}
            entry["by_class"] = {str(k): paired(a[y == k], b[y == k], y[y == k])
                                 for k in sorted(set(y.tolist()))}
        comparisons[split] = entry
    result = {"status": "completed", "matched_start_order_budget": True,
              "comparisons": comparisons,
              "readout_gain_norm": weighted["final"]["readout_gain_norm"],
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), *paths]},
              "scope": "One seed; train-file speakers 3/6 held out; official test untouched; descriptive pairing"}
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
