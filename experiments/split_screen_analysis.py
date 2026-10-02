"""Validate and summarize the frozen split-event screen; no fitting or holdouts."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_aws_matrix_recovery import validate_result

ROOT = Path(__file__).resolve().parents[1]


def paired_accuracy(first, second, paired_timing=False, draws=10000):
    """Compare the same populations; timing's two members stay one cluster."""
    a = np.asarray(first['episode_accuracy'], dtype=float)
    b = np.asarray(second['episode_accuracy'], dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a):
        raise ValueError('Identical nonempty population arrays required')
    if paired_timing:
        if len(a) % 2:
            raise ValueError('Whole timing pairs required')
        a, b = a.reshape(-1, 2).mean(1), b.reshape(-1, 2).mean(1)
    delta = a-b
    if len(delta) < 2:
        raise ValueError('At least two independent populations required')
    samples = np.random.default_rng(719).choice(delta, (draws, len(delta))).mean(1)
    return dict(accuracy_gain_pp=float(delta.mean()*100),
                cluster_bootstrap_95_pp=(np.quantile(samples, (.025, .975))*100).tolist(),
                independent_clusters=len(delta),
                scope='Exploratory paired population bootstrap, conditional on selected seed6 checkpoints; '
                      'development selection and multiple screen comparisons are not corrected')


def analyze(manifest):
    plan = json.loads(manifest.read_text())
    rows = {}
    for job in plan['jobs']:
        if job['stage'] != 'pilot':
            continue
        row = validate_result(ROOT, job)
        a, w = row['args'], row['work']
        if (a['fit_targets'], a['dev_targets'], a['epochs'], a['seed']) != (128, 256, 4, 6):
            raise ValueError('Expected frozen screening budget')
        if w['fitting_query_targets'] != 512 or w['fitting_events'] != 2048:
            raise ValueError('Incomplete fitting work')
        rows[job['variant']] = row
    if len(rows) != 11:
        raise ValueError('All eleven completed pilots required')

    def contrast(first, second, paired=False):
        a, b = rows[first], rows[second]
        if a['data_sha256'] != b['data_sha256']:
            raise ValueError('Paired contrasts require identical data')
        result = paired_accuracy(a['final']['dev'], b['final']['dev'], paired)
        result.update(first=first, second=second,
            first_whole_fit_gflops=a['work']['total_training_unit_special_flops']/1e9,
            second_whole_fit_gflops=b['work']['total_training_unit_special_flops']/1e9,
            first_parameters=a['parameters'], second_parameters=b['parameters'],
            nll_first=a['final']['dev']['nll'], nll_second=b['final']['dev']['nll'])
        return result

    contrasts = [contrast(f'order_S{s}_shared_P0', f'order_S{s}_private_P0') for s in (4, 16)]
    contrasts.append(contrast('paired_timing_S4_private_P0_observed',
                              'paired_timing_S4_private_P0_rank', True))
    return dict(status='completed', manifest=str(manifest.relative_to(ROOT)),
        completed_pilots=len(rows), contrasts=contrasts,
        cells={name:dict(accuracy=row['final']['dev']['accuracy'], nll=row['final']['dev']['nll'],
                        accuracy_interval=row['final']['dev']['episode_bootstrap_95_percent_interval'],
                        parameters=row['parameters'], work=row['work']['total_training_unit_special_flops'],
                        state_slots=row['work']['available_receivers'],
                        updates=row['work']['selected_updates_per_event'],
                        scores=row['work']['key_scores_per_event']) for name,row in rows.items()},
        scope='Single-seed synthetic development screen; no independent confirmation, '
              'time-aware dense comparator, real-data supremacy or physical-energy measurement')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Never overwrite completed analysis')
    result=analyze(a.manifest.resolve())
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['contrasts'],indent=2))
