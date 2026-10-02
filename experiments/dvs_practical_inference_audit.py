"""Frozen, causal per-prefix CPU inference against all saved gesture controls."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import sys
import time
import warnings

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_native_benchmark as N


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(probabilities, labels):
    p = np.asarray(probabilities, dtype=np.float64)
    p /= p.sum(1, keepdims=True)
    if np.any(p[np.arange(len(labels)), labels] <= 0):
        raise ValueError('Positive target probabilities required')
    return dict(accuracy=float(np.mean(p.argmax(1) == labels)),
        nll=float(-np.log(p[np.arange(len(labels)), labels]).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--native', required=True)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args(); start = time.perf_counter(); torch.set_num_threads(1)
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists() or not 1 <= a.repeats <= 5:
        raise ValueError('Unused bounded audit required')
    path = ROOT / a.native; native = json.loads(path.read_text())
    if native['status'] != 'completed':
        raise ValueError('Completed native fit required')
    for name, sha in native['source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Changed source: ' + name)
    args = argparse.Namespace(**native['args']); fit, dev, info = N.load(args)
    if info != native['data'] or len(fit) != 984 or len(dev) != 192:
        raise ValueError('Unchanged full comparison required')
    control_path = ROOT / args.controls; controls = json.loads(control_path.read_text())
    artifact = ROOT / controls['model_artifact']
    if digest(artifact) != controls['model_artifact_sha256']:
        raise ValueError('Changed fitted controls')
    saved = joblib.load(artifact)
    parent = json.loads((ROOT / args.data).read_text())
    data_path = ROOT / parent['data_artifact']
    z = np.load(data_path, allow_pickle=False); raw = z['dev_counts']; labels = z['dev_labels']
    if [x['target'] for x in dev] != labels.tolist():
        raise ValueError('Changed target order')
    checkpoint = path.with_suffix('.progress.pt')
    ck = torch.load(checkpoint, weights_only=False)
    if ck['source_sha256'] != native['source_sha256'] or ck['data'] != native['data']:
        raise ValueError('Changed checkpoint source/data')
    if ck['cursor']['epoch'] != args.epochs + 1 or ck['result']['selected_epoch'] != native['selected_epoch']:
        raise ValueError('Completed selected checkpoint required')
    model = N.make_model(args); model.load_state_dict(ck['best_state']); model.eval()
    rows = []

    def measure(name, predict, reference, artifact_object, tensor_bytes=None):
        # Same sequential causal calls for every model. Warmup is reported and
        # excluded from steady-state timings; no target enters prediction.
        warm_start = time.perf_counter(); predict(raw[0]); warm_wall = time.perf_counter() - warm_start
        times = []; all_probabilities = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for _ in range(a.repeats):
                begin = time.perf_counter()
                probabilities = [predict(counts) for counts in raw]
                times.append(time.perf_counter() - begin)
                all_probabilities.append(np.asarray(probabilities))
        expected = np.asarray(reference['probabilities'])
        for probabilities in all_probabilities:
            if not np.allclose(probabilities, expected, rtol=1e-5, atol=1e-6):
                raise ValueError('Frozen probabilities changed: ' + name)
        for probabilities in all_probabilities[1:]:
            np.testing.assert_array_equal(probabilities, all_probabilities[0])
        quality = score(all_probabilities[0], labels)
        if abs(quality['nll'] - reference['nll']) > 1e-5 or quality['accuracy'] != reference['accuracy']:
            raise ValueError('Changed frozen quality: ' + name)
        buffer = io.BytesIO(); joblib.dump(artifact_object, buffer, compress=0)
        rows.append(dict(arm=name, quality=quality, targets=len(raw), repeats=a.repeats,
            sequential_prefix_wall_s=times, median_wall_ms_per_prefix=float(np.median(times)) * 1000 / len(raw),
            warmup_wall_s=warm_wall, uncompressed_joblib_bytes=len(buffer.getvalue()),
            model_parameter_tensor_bytes=tensor_bytes,
            warnings=sorted(set(str(x.message) for x in caught)),
            scope='CPU one-thread sequential prefix calls, fit-only transform included; complete observed count packet available at1s. Raw coalescing, loading and warmup separate. Joblib size is storage, not process RSS or traffic.'))

    def count_predict(counts):
        values = (counts.reshape(1, -1) @ saved['count_log_values'].T + saved['count_log_prior']) / saved['count_temperature']
        values -= values.max(); values = np.exp(values); return (values / values.sum())[0]
    by = {x['arm']: x for x in controls['rows']}
    count_object = {k: saved[k] for k in ('count_log_values', 'count_log_prior', 'count_temperature')}
    measure('time_binned_naive_bayes', count_predict, by['time_binned_naive_bayes']['final'], count_object)
    for name, control in saved['models'].items():
        def predict(counts, control=control):
            transformed = (np.log1p(counts.reshape(1, -1)) - saved['center']) / saved['scale']
            return control.predict_proba(transformed)[0]
        measure(name, predict, by[name]['final'], dict(model=control, center=saved['center'], scale=saved['scale']))
    with torch.no_grad():
        def predict_native(counts):
            events = N.encode(counts, saved['center'], saved['scale'])
            logits, _ = N.predict(model, dict(events=events), 314159, False)
            return logits.softmax(-1).numpy().copy()
        model_object = dict(state_dict=model.state_dict(), settings={k: v for k, v in vars(args).items() if k not in ('tag', 'resume', 'stop_after_updates')},
            center=saved['center'], scale=saved['scale'])
        measure('native', predict_native, native['final'], model_object,
            sum(x.numel() * x.element_size() for x in model.parameters()))
    for name, sha in native['source_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError('Source changed during audit')
    result = dict(status='completed', args=vars(a), common_sequential_inference_ledger=rows,
        native_result_sha256=digest(path), native_checkpoint_sha256=digest(checkpoint),
        control_result_sha256=digest(control_path), control_artifact_sha256=digest(artifact),
        raw_preprocessing_wall_s=controls['preprocessing_wall_s'],
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, wall_s=time.perf_counter() - start,
        source_sha256={**N.sources(), 'experiments/dvs_practical_inference_audit.py': digest(__file__)},
        scope='Same frozen DEVELOPMENT probabilities, no fitting or test access. CPU emulator latency/storage only, not measured energy or raw-spike event hardware. Three deterministic repeats measure timing variation, not model uncertainty. No independent practical advantage claim.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
