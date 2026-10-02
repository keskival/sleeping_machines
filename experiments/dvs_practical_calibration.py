"""Bounded, train-file-only DVS calibration before a new integrated fit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import warnings
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from e120_dvs_adapter import dvs


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frames(example):
    """Same observed 50ms packets; no labels or fitted evidence enter input."""
    p = example.prefix; p.validate()
    if example.evidence is not None: raise ValueError('Observed inputs only')
    if abs(p.cutoff - 1.) > 1e-9: raise ValueError('First-second protocol required')
    mask = p.channels < 32
    bins = np.rint(p.times[mask] / .05).astype(np.int64) - 1
    if np.any((bins < 0) | (bins >= 20)): raise ValueError('Packet outside observation')
    if not np.allclose(p.times[mask], (bins + 1) * .05, rtol=0, atol=1e-9):
        raise ValueError('Actual packet closure times required')
    out = np.zeros((20, 32), dtype=np.float64)
    np.add.at(out, (bins, p.channels[mask]), p.counts[mask])
    if out.sum() != p.counts[mask].sum(): raise ValueError('Lost observed events')
    return out


def sources():
    names = ['experiments/dvs_practical_calibration.py', 'experiments/e120_dvs_adapter.py',
             'experiments/e120_shared_tasks.py', 'sleeping_machines/event_query.py']
    return {name: digest(ROOT / name) for name in names}


def data(a, output):
    started = time.perf_counter(); task = dvs(a.fit, a.dev, a.seed)
    if task.protocol['official_test_read']: raise ValueError('Train-file data only')
    assert not set(x.identity for x in task.fit) & set(x.identity for x in task.dev)
    arrays = {}
    for name, rows in [('fit', task.fit), ('dev', task.dev)]:
        arrays[name + '_counts'] = np.stack([frames(x) for x in rows])
        arrays[name + '_labels'] = np.array([x.label for x in rows], dtype=np.int64)
        arrays[name + '_ids'] = np.array([x.identity for x in rows])
    x = task.fit[0]; changed = copy.copy(x); changed.label = (x.label + 1) % 11
    np.testing.assert_array_equal(frames(x), frames(changed))
    np.testing.assert_array_equal(frames(x), frames(copy.deepcopy(x)))
    path = output.with_suffix('.data.npz')
    if path.exists(): raise ValueError('Preserve previous data artifact')
    np.savez_compressed(path, **arrays)
    return dict(data_artifact=str(path.relative_to(ROOT)), data_sha256=digest(path), protocol=task.protocol,
        fitting_targets=len(task.fit), development_targets=len(task.dev), frames_per_prefix=20, channels=32,
        raw_count_totals={name: float(arrays[name + '_counts'].sum()) for name in ('fit', 'dev')},
        label_and_evidence_input_contracts_passed=True, preprocessing_wall_s=time.perf_counter() - started,
        scope='Existing causal first-second 4x4/polarity/50ms adapter, users1..19 fit/users20..23 development; '
              'same observed packet counts, no fitted evidence or labels in inputs, no official-test access. '
              'Dense frame tensor is common representation, not a claim of raw-spike execution.')


def controls(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    import sklearn
    path = ROOT / a.data; parent = json.loads(path.read_text())
    if parent['status'] != 'completed' or not parent['label_and_evidence_input_contracts_passed']:
        raise ValueError('Completed input contracts required')
    artifact = ROOT / parent['data_artifact']
    if digest(artifact) != parent['data_sha256']: raise ValueError('Changed data artifact')
    for name, sha in parent['source_sha256'].items():
        if digest(ROOT / name) != sha: raise ValueError('Changed data source')
    z = np.load(artifact, allow_pickle=False)
    raw_fit = z['fit_counts'].reshape(len(z['fit_labels']), -1)
    raw_dev = z['dev_counts'].reshape(len(z['dev_labels']), -1)
    yfit, ydev = z['fit_labels'], z['dev_labels']; dimension = raw_fit.shape[1]
    logged = np.log1p(raw_fit); center = logged.mean(0); scale = np.maximum(logged.std(0), .5)
    xfit = (logged - center) / scale; xdev = (np.log1p(raw_dev) - center) / scale
    rows = []; models = {}
    def from_logits(logits, labels):
        shifted = logits - logits.max(1, keepdims=True)
        logs = shifted - np.log(np.exp(shifted).sum(1, keepdims=True))
        return np.exp(logs), float(-logs[np.arange(len(labels)), labels].mean())
    def score(probabilities, exact_nll=None):
        p = probabilities / probabilities.sum(1, keepdims=True)
        if exact_nll is None:
            if np.any(p[np.arange(len(ydev)), ydev] <= 0): raise ValueError('Zero predicted target probability')
            exact_nll = float(-np.log(p[np.arange(len(ydev)), ydev]).mean())
        return dict(targets=len(ydev), accuracy=float(np.mean(p.argmax(1) == ydev)),
            nll=exact_nll, probabilities=p.tolist())
    # Fit-only calibration prevents treating the correlated camera counts as
    # thousands of independent label observations with huge overconfidence.
    start = time.perf_counter()
    def count_fit(indices, alpha):
        totals = np.full((11, dimension), alpha); class_counts = np.ones(11)
        for i in indices: totals[yfit[i]] += raw_fit[i]; class_counts[yfit[i]] += 1
        return np.log(totals / totals.sum(1, keepdims=True)), np.log(class_counts / class_counts.sum())
    folds = list(StratifiedKFold(3, shuffle=True, random_state=a.seed).split(raw_fit, yfit))
    count_grid = []; temperatures = np.logspace(0, 4, 9)
    for alpha in (.1, 1., 10., 100.):
        logits = np.empty((len(yfit), 11))
        for training, held in folds:
            values, prior = count_fit(training, alpha)
            logits[held] = raw_fit[held] @ values.T + prior
        for temperature in temperatures:
            _, nll = from_logits(logits / temperature, yfit)
            count_grid.append(dict(alpha=alpha, temperature=float(temperature), fitting_oof_nll=nll))
    count_choice = min(count_grid, key=lambda row: row['fitting_oof_nll'])
    log_values, log_prior = count_fit(np.arange(len(yfit)), count_choice['alpha'])
    fit_wall = time.perf_counter() - start; start = time.perf_counter()
    probabilities, nll = from_logits((raw_dev @ log_values.T + log_prior) / count_choice['temperature'], ydev)
    rows.append(dict(arm='time_binned_naive_bayes', final=score(probabilities, nll), fitting_wall_s=fit_wall,
        inference_wall_s=time.perf_counter() - start, calibration_grid=count_grid, selected_from_fitting_only=count_choice,
        fitting_scope='Class-conditional observed packet counts/all640 bins;3-fold FIT-ONLY alpha/temperature calibration,12fold fits+1full fit.'))
    # Fixed grid declared before seeing this development set; keep every arm.
    for strength in (.1, 1., 10.):
        models[f'linear_C{strength:g}'] = LogisticRegression(C=strength, max_iter=1000, random_state=a.seed)
    for strength in (1., 10.):
        for multiplier in (.25, 1., 4.):
            gamma = multiplier / (dimension * xfit.var())
            models[f'rbf_C{strength:g}_g{multiplier:g}'] = SVC(C=strength, gamma=gamma, probability=True, random_state=a.seed)
    for name, model in models.items():
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); model.fit(xfit, yfit)
        fit_wall = time.perf_counter() - start; start = time.perf_counter(); probabilities = model.predict_proba(xdev)
        if model.classes_.tolist() != list(range(11)): raise ValueError('All class alphabet required')
        inference_wall = time.perf_counter() - start
        row = dict(arm=name, final=score(probabilities), fitting_wall_s=fit_wall, inference_wall_s=inference_wall,
                   iterations=np.asarray(model.n_iter_).tolist(), warnings=[str(w.message) for w in caught])
        if hasattr(model, 'support_'): row['support_vectors'] = len(model.support_)
        rows.append(row); print(json.dumps(dict(arm=name, accuracy=row['final']['accuracy'], nll=row['final']['nll'])), flush=True)
    import joblib
    model_path = path.with_name(a.tag + '.models.joblib')
    if model_path.exists(): raise ValueError('Preserve old models')
    joblib.dump(dict(models=models, center=center, scale=scale, count_log_values=log_values, count_log_prior=log_prior,
                     count_temperature=count_choice['temperature']), model_path)
    selected = min(rows[1:], key=lambda r: r['final']['nll']); count = rows[0]
    gain = selected['final']['accuracy'] - count['final']['accuracy']
    return dict(data_result=a.data, data_result_sha256=digest(path), model_artifact=str(model_path.relative_to(ROOT)),
        model_artifact_sha256=digest(model_path), sklearn_version=sklearn.__version__, protocol=parent['protocol'],
        fitting_targets=len(yfit), development_targets=len(ydev), rows=rows, selected_by_dev_nll=selected['arm'],
        selected_control_accuracy_gain_over_counts=gain,
        development_region_admission_passed=gain >= .05 and max(row['final']['accuracy'] for row in rows) < .95,
        preprocessing_wall_s=parent['preprocessing_wall_s'],
        scope='Strong fixed linear/RBF grids against time-aware class counts; selection by development NLL only. '
              'Not independent confirmation or architectural advantage. All tuning/Platt internal CV in measured fit wall; '
              'third-party solver FLOPs unavailable, not zero. Common raw preprocessing charged separately; '
              'no Transformer/LSTM training or official test. Admission requires >=5pp learning gain and >=5pp accuracy headroom.')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--tag', required=True)
    p.add_argument('--stage', choices=('data', 'controls'), required=True); p.add_argument('--data')
    p.add_argument('--fit', type=int, default=984); p.add_argument('--dev', type=int, default=192)
    p.add_argument('--seed', type=int, default=6); a = p.parse_args()
    out = ROOT / 'experiments/results/dvs_calibration' / (a.tag + '.json'); out.parent.mkdir(exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists(): raise ValueError('Unused plain tag required')
    start = time.perf_counter(); result = data(a, out) if a.stage == 'data' else controls(a)
    result.update(status='completed', args=vars(a), source_sha256=sources(), wall_s=time.perf_counter() - start,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__': main()
