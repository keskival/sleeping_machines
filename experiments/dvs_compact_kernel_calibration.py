"""Strong bounded-storage gesture controls before claiming native storage advantage."""
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
from sklearn.cluster import KMeans
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
import sklearn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def feature_map(transformed, model):
    if model['kind'] == 'nystroem':
        return model['basis'].transform(transformed)
    centers = model['centers']
    squared = np.maximum((transformed * transformed).sum(1, keepdims=True)
        + (centers * centers).sum(1)[None] - 2 * transformed @ centers.T, 0.)
    return np.exp(-model['gamma'] * squared)


def predict(counts, model):
    transformed = ((np.log1p(counts.reshape(1, -1)) - model['center']) / model['scale']).astype(np.float32)
    return model['decoder'].predict_proba(feature_map(transformed, model))[0]


def score(probabilities, labels):
    p = np.asarray(probabilities); target = p[np.arange(len(labels)), labels]
    if np.any(target <= 0): raise ValueError('Positive target probabilities required')
    return dict(targets=len(labels), accuracy=float(np.mean(p.argmax(1) == labels)),
        nll=float(-np.log(target).mean()), probabilities=p.tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True); parser.add_argument('--controls', required=True)
    parser.add_argument('--audit', required=True)
    a = parser.parse_args(); started = time.perf_counter()
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists(): raise ValueError('Unused plain tag required')
    control_path = ROOT / a.controls; control = json.loads(control_path.read_text())
    audit_path = ROOT / a.audit; audit = json.loads(audit_path.read_text())
    if control['status'] != audit['status'] or control['status'] != 'completed': raise ValueError('Completed controls/audit required')
    if audit['control_result_sha256'] != sha(control_path): raise ValueError('Changed comparator')
    native = next(x for x in audit['common_sequential_inference_ledger'] if x['arm'] == 'native')
    budget = native['uncompressed_joblib_bytes']
    data_path = ROOT / control['data_result']; parent = json.loads(data_path.read_text())
    if sha(data_path) != control['data_result_sha256'] or parent['protocol']['official_test_read']: raise ValueError('Unchanged causal data required')
    artifact = ROOT / parent['data_artifact']
    if sha(artifact) != parent['data_sha256']: raise ValueError('Changed packets')
    for name, digest in control['source_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Changed calibration source')
    z = np.load(artifact, allow_pickle=False); raw_fit = z['fit_counts'].reshape(984, -1)
    logged = np.log1p(raw_fit); center = logged.mean(0); scale = np.maximum(logged.std(0), .5)
    xfit = ((logged - center) / scale).astype(np.float32)
    xdev = ((np.log1p(z['dev_counts'].reshape(192, -1)) - center) / scale).astype(np.float32)
    labels, dev_labels = z['fit_labels'], z['dev_labels']; dimensions = xfit.shape[1]
    rows = []; models = {}; shared_bases = []; preprocessing_started = time.perf_counter()
    # Fixed fitting-only bases: random landmarks and supervised class mixtures.
    for k in (1, 2, 3, 4):
        begin = time.perf_counter(); centers = []
        for label in range(11):
            values = xfit[labels == label]
            centers.extend([values.mean(0)] if k == 1 else KMeans(n_clusters=k, n_init=5,
                max_iter=100, random_state=681).fit(values).cluster_centers_)
        shared_bases.append(dict(kind='class_prototype', centers=np.asarray(centers, dtype=np.float32),
            components=11*k, basis_fitting_wall_s=time.perf_counter() - begin))
    prototype_basis_wall = time.perf_counter() - preprocessing_started
    for components in (8, 16, 24, 32):
        shared_bases.append(dict(kind='nystroem', components=components, basis_fitting_wall_s=0.))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for definition in shared_bases:
            for multiplier in (.25, 1., 4.):
                gamma = multiplier / (dimensions * xfit.var())
                base = dict(definition, gamma=float(gamma), center=center, scale=scale)
                begin = time.perf_counter()
                if base['kind'] == 'nystroem':
                    base['basis'] = Nystroem(kernel='rbf', gamma=gamma,
                        n_components=base['components'], random_state=681).fit(xfit)
                features = feature_map(xfit, base); dev_features = feature_map(xdev, base)
                feature_wall = time.perf_counter() - begin
                for strength in (.1, 1., 10.):
                    name = f"{base['kind']}_m{base['components']}_g{multiplier:g}_C{strength:g}"
                    decoder = LogisticRegression(C=strength, max_iter=1000, random_state=681)
                    begin = time.perf_counter(); decoder.fit(features, labels); fit_wall = time.perf_counter() - begin
                    model = dict(base, decoder=decoder); probabilities = decoder.predict_proba(dev_features)
                    quality = score(probabilities, dev_labels)
                    buffer = io.BytesIO(); joblib.dump(model, buffer, compress=0); size = len(buffer.getvalue())
                    begin = time.perf_counter(); sequential = np.array([predict(counts, model) for counts in z['dev_counts']])
                    inference_wall = time.perf_counter() - begin
                    np.testing.assert_allclose(sequential, probabilities, rtol=1e-4, atol=1e-5)
                    rows.append(dict(arm=name, final=quality, components=base['components'],
                        within_native_storage_budget=size <= budget, uncompressed_joblib_bytes=size,
                        head_fitting_wall_s=fit_wall, shared_feature_wall_s=feature_wall,
                        shared_prototype_basis_wall_s=base['basis_fitting_wall_s'],
                        sequential_ms_per_prefix=1000*inference_wall/len(dev_labels)))
                    models[name] = model
    eligible = [x for x in rows if x['within_native_storage_budget']]
    if not eligible: raise ValueError('At least one bounded-storage control required')
    selected = min(eligible, key=lambda x: x['final']['nll'])
    artifact_out = out.with_suffix('.models.joblib'); joblib.dump(models, artifact_out)
    result = dict(status='completed', args=vars(a), rows=rows, native_storage_budget_bytes=budget,
        selected_within_budget_by_dev_nll=selected['arm'],
        maximum_within_budget_accuracy=max(x['final']['accuracy'] for x in eligible),
        native_development_quality=native['quality'], selected_compact_development_quality=selected['final'],
        native_dominates_selected_compact_development_quality=native['quality']['accuracy'] >= selected['final']['accuracy'] and native['quality']['nll'] < selected['final']['nll'],
        prototype_basis_total_wall_s=prototype_basis_wall, controls_sha256=sha(control_path),
        inference_audit_sha256=sha(audit_path), model_artifact=str(artifact_out.relative_to(ROOT)),
        model_artifact_sha256=sha(artifact_out), sklearn_version=sklearn.__version__,
        protocol=parent['protocol'], preprocessing_wall_s=control['preprocessing_wall_s'],
        source_sha256={'experiments/dvs_compact_kernel_calibration.py':sha(__file__)},
        warnings=sorted(set(str(x.message) for x in caught)),
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='72 fixed compact kernel/class-prototype controls, same984/192 observed packets. Random8/16/24/32 Nyström landmarks or1/2/3/4 class prototypes; fitting-only transform/bases,3 gamma and3 logistic C values. Storage budget is actual original native joblib artifact size, same uncompressed serialization including required transform. Every cell retained, dev-only selection, no independent advantage claim or test access. Shared basis/head/all tuning costs paid in total wall; solver FLOPs unknown, not zero. One timing pass per control is exploratory, not a repeated latency claim.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
