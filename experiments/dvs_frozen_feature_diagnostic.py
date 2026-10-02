"""Diagnose retained gesture information with frozen initial/fitted native encoders."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import warnings

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
import dvs_native_benchmark as N


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nll(probabilities, labels):
    p = probabilities[np.arange(len(labels)), labels]
    if np.any(p <= 0): raise ValueError('Positive target probabilities required')
    return float(-np.log(p).mean())


@torch.no_grad()
def extract(model, rows):
    last = []; features = []; probabilities = []
    def hook(module, inputs):
        last[:] = [inputs[0].detach().clone()]
    handle = model.head.register_forward_pre_hook(hook)
    try:
        for row in rows:
            logits, _ = N.predict(model, row, 314159, False)
            features.append(last[0].numpy().copy()); probabilities.append(logits.softmax(-1).numpy().copy())
            torch.testing.assert_close(model.head(last[0]), logits, rtol=0, atol=0)
    finally:
        handle.remove()
    return np.asarray(features), np.asarray(probabilities)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--native', required=True); p.add_argument('--analysis', required=True)
    a = p.parse_args(); started = time.perf_counter(); torch.set_num_threads(1)
    out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.json')
    if Path(a.tag).name != a.tag or out.exists(): raise ValueError('Unused plain tag required')
    native_path = ROOT / a.native; r = json.loads(native_path.read_text())
    analysis = json.loads((ROOT / a.analysis).read_text())
    if r['status'] != analysis['status'] or r['status'] != 'completed': raise ValueError('Completed parent comparison required')
    if analysis['inputs'][a.native] != sha(native_path): raise ValueError('Changed parent analysis')
    if analysis['native_dominates_selected_control_in_development_quality']:
        raise ValueError('Failure diagnostic requires completed failure against strong control')
    args = argparse.Namespace(**r['args']); fit, dev, info = N.load(args)
    if info != r['data'] or len(fit) != 984 or len(dev) != 192: raise ValueError('Unchanged full protocol required')
    for source, digest in r['source_sha256'].items():
        if sha(ROOT / source) != digest: raise ValueError('Changed native source')
    ckpath = native_path.with_suffix('.progress.pt'); ck = torch.load(ckpath, weights_only=False)
    if ck['source_sha256'] != r['source_sha256'] or ck['data'] != r['data']: raise ValueError('Changed checkpoint')
    labels = np.array([x['target'] for x in fit]); dev_labels = np.array([x['target'] for x in dev])
    folds = list(StratifiedKFold(3, shuffle=True, random_state=681).split(np.zeros(len(fit)), labels))
    configurations = [dict(kind='linear', C=c, gamma=None) for c in (.1, 1., 10.)]
    configurations += [dict(kind='rbf', C=c, gamma=g) for c in (1., 10.) for g in (.25, 1., 4.)]
    rows = []; artifacts = {}
    for arm in ('initial', 'selected'):
        model = N.make_model(args)
        if arm == 'selected': model.load_state_dict(ck['best_state'])
        model.eval(); before = {name: x.clone() for name, x in model.state_dict().items()}
        begin = time.perf_counter(); xfit, _ = extract(model, fit); xdev, original = extract(model, dev)
        replay_wall = time.perf_counter() - begin
        reference = r['initial_dev'] if arm == 'initial' else r['final']
        if not np.allclose(original, reference['probabilities'], rtol=1e-5, atol=1e-6): raise ValueError('Changed original inference')
        candidates = []; begin = time.perf_counter()
        def transform(training, evaluation):
            center = training.mean(0); scale = np.maximum(training.std(0), .5)
            return (training - center) / scale, (evaluation - center) / scale, center, scale
        def classifier(config, transformed):
            if config['kind'] == 'linear':
                return LogisticRegression(C=config['C'], max_iter=1000, random_state=681)
            return SVC(C=config['C'], gamma=config['gamma'] / (transformed.shape[1] * max(float(transformed.var()), 1e-12)),
                probability=True, random_state=681)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            for config in configurations:
                predictions = np.empty((len(fit), 11))
                for training, held in folds:
                    xt, xv, _, _ = transform(xfit[training], xfit[held]); decoder = classifier(config, xt)
                    decoder.fit(xt, labels[training]); predictions[held] = decoder.predict_proba(xv)
                candidates.append(dict(configuration=config, fitting_decoder_cv_nll=nll(predictions, labels)))
            chosen = min(candidates, key=lambda x: x['fitting_decoder_cv_nll'])
            xt, xv, center, scale = transform(xfit, xdev); decoder = classifier(chosen['configuration'], xt)
            decoder.fit(xt, labels); probabilities = decoder.predict_proba(xv)
        fitting_wall = time.perf_counter() - begin
        for name, x in model.state_dict().items(): torch.testing.assert_close(x, before[name], rtol=0, atol=0)
        quality = dict(accuracy=float(np.mean(probabilities.argmax(1) == dev_labels)), nll=nll(probabilities, dev_labels),
            probabilities=probabilities.tolist())
        rows.append(dict(encoder=arm, feature_dimension=xfit.shape[1], original_development=dict(accuracy=reference['accuracy'], nll=reference['nll']),
            frozen_decoder_development=quality, fit_only_selected_decoder=chosen, all_fit_cv_candidates=candidates,
            replay_wall_s=replay_wall, decoder_grid_wall_s=fitting_wall,
            original_encoder_fit_gflops_estimate=r['work']['whole_fit_unit_special_flops_estimate']/1e9 if arm == 'selected' else 0.,
            original_encoder_workflow_wall_s=r['wall_s'] if arm == 'selected' else 0.,
            warnings=sorted(set(str(x.message) for x in caught)), state_and_parameters_preserved=True))
        artifacts[arm] = dict(decoder=decoder, feature_center=center, feature_scale=scale)
        print(json.dumps(dict(encoder=arm, accuracy=quality['accuracy'], nll=quality['nll'])), flush=True)
    artifact = out.with_suffix('.heads.joblib'); joblib.dump(artifacts, artifact)
    for source, digest in r['source_sha256'].items():
        if sha(ROOT / source) != digest: raise ValueError('Source changed during diagnostic')
    result = dict(status='completed', args=vars(a), rows=rows, native_result_sha256=sha(native_path),
        native_checkpoint_sha256=sha(ckpath), parent_analysis_sha256=sha(ROOT/a.analysis),
        selected_heads_artifact=str(artifact.relative_to(ROOT)), selected_heads_sha256=sha(artifact),
        source_sha256={**N.sources(), 'experiments/dvs_frozen_feature_diagnostic.py': sha(__file__)},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Frozen representation/readout diagnostic after a completed native failure. Initial reservoir and selected encoder use the same replay/head-selection protocol. Decoder hyperparameters chosen by3-fold FIT-only NLL; selected encoder previously trained on all fitting labels, so conditional decoder CV is not unbiased end-to-end validation. All native fitting/replay/head-grid costs retained; solver FLOPs unknown. No core substitution, retuning on dev, official-test use, useful-depth or practical advantage claim.')
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
