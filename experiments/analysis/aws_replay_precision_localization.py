"""Read-only analysis of saved gradients; performs no model execution or fitting."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/results/diagnostics/local_language_precision_audit_20261003T015800Z.vectors.npz'


def analyze():
    rows = []
    with np.load(SOURCE, allow_pickle=False) as saved:
        for family in ('private', 'depth'):
            for kind in ('original', 'reuse'):
                prefix = f'{family}_float32_{kind}'
                names = saved[prefix + '_parameter_names']
                sizes = saved[prefix + '_parameter_sizes']
                candidate = saved[prefix + '_gradients']
                reference = saved[f'{family}_float64_{kind}_gradients']
                delta = candidate - reference
                total = float(np.dot(delta, delta))
                offset = 0
                parameters = []
                for name, size in zip(names, sizes):
                    end = offset + int(size)
                    d, r = delta[offset:end], reference[offset:end]
                    squared = float(np.dot(d, d))
                    parameters.append(dict(name=str(name), coordinates=int(size),
                        squared_error_share=squared / total if total else 0.,
                        max_absolute_error=float(np.max(np.abs(d))),
                        failed_coordinates=int(np.count_nonzero(np.abs(d) > 3e-6 + 3e-4 * np.abs(r)))))
                    offset = end
                assert offset == len(reference)
                rows.append(dict(family=family, estimator=kind,
                    relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(reference)),
                    failed_coordinates=sum(p['failed_coordinates'] for p in parameters),
                    parameters=sorted(parameters, key=lambda p: p['squared_error_share'], reverse=True)))
    return dict(scope='Secondary analysis of already completed gradient vectors; no training, optimizer, new forward/backward or quality evidence.',
        source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        coordinate_gate=dict(rtol=3e-4, atol=3e-6), rows=rows)


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2, allow_nan=False))
