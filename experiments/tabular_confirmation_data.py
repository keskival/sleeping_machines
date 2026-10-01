"""Reserved banknote rows; normalize using the already-frozen fitting scaler."""
import hashlib
import io

import numpy as np

from experiments.native_tabular_data import load, ROOT


def confirmation_rows(protocol):
    if protocol['dataset'] != 'banknote':
        raise ValueError('This confirmation protocol is banknote only')
    raw = (ROOT/'data/tabular/data_banknote_authentication.txt').read_bytes()
    if hashlib.sha256(raw).hexdigest() != protocol['raw_sha256']:
        raise ValueError('Dataset bytes changed')
    rows = np.loadtxt(io.StringIO(raw.decode()), delimiter=',')
    ids = np.asarray(protocol['reserved_test_indices'], dtype=int)
    if (len(ids) == 0 or len(set(ids)) != len(ids)
            or set(ids) & set(protocol['fit_indices'] + protocol['dev_indices'])):
        raise ValueError('Nonempty disjoint reserved rows required')
    mean = np.asarray(protocol['preprocessing_mean'])
    scale = np.asarray(protocol['preprocessing_std'])
    _, groups = np.unique(rows[:, :-1], axis=0, return_inverse=True)
    return dict(x=(rows[ids, :-1]-mean)/scale, y=rows[ids, -1], indices=ids.tolist(),
                groups=groups[ids].tolist(),
                sha256=hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()
                    +np.asarray(rows[ids],dtype='<f8').tobytes()).hexdigest())


def prediction_rows(logits, targets, indices):
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=int)
    if (logits.shape != (len(targets), 2) or len(indices) != len(targets)
            or not np.isfinite(logits).all() or not np.isin(targets,(0,1)).all()):
        raise ValueError('Finite two-class predictions on matching rows required')
    logp = logits - np.logaddexp(logits[:,0],logits[:,1])[:,None]
    losses = -logp[np.arange(len(targets)), targets]
    correct = logits.argmax(1) == targets
    return dict(indices=list(indices), targets=targets.tolist(), log_probabilities=logp.tolist(),
                losses=losses.tolist(), correct=correct.tolist())


def contracts():
    data = load('banknote',32,32)
    test = confirmation_rows(data['protocol'])
    assert set(test['indices']).isdisjoint(data['protocol']['fit_indices'])
    assert set(test['indices']).isdisjoint(data['protocol']['dev_indices'])
    return dict(reserved_rows_disjoint=True, frozen_training_scaler_on_test=True,
                data_bytes_verified=True, explicit_test_row_identity=True)
