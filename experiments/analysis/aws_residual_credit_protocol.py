"""Prepare a source-bound trained-state residual replay diagnostic; no execution."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__ == '__main__':
    candidates=[]
    for path in (ROOT/'experiments/results/aws_language_progress').glob('*private_replay_10m_s7_milestone*.json'):
        record=json.loads(path.read_text())
        if record['trained_targets']>=1_000_000:
            candidates.append((record['trained_targets'],path,record))
    targets,path,record=min(candidates)  # Fixed earliest trained admission state, not score selection.
    checkpoint=ROOT/record['checkpoint']
    assert digest(checkpoint)==record['checkpoint_sha256']
    assert record['args']['receiver_sharing']=='private'
    source_names=['experiments/aws_language_winner_reuse.py',
                  'experiments/theory/aws_20261003_residual_counterfactual_credit.md',
                  'experiments/analysis/aws_residual_credit_algebra.py',
                  'experiments/analysis/aws_residual_credit_protocol.py']
    result=dict(status='prepared_not_executed',checkpoint=record['checkpoint'],
        checkpoint_sha256=record['checkpoint_sha256'],trained_targets=targets,
        checkpoint_metadata=str(path.relative_to(ROOT)),checkpoint_metadata_sha256=digest(path),
        producer_source_sha256=record['source_sha256'],
        protocol_source_sha256={name:digest(ROOT/name) for name in source_names},
        scope='Admission protocol only. No trained residuals, kernel contracts, optimizer forks or work measurements produced.',
        retained='Factual temporal races/clocks, private persistent writes, separate keys/values, all counterfactual route support; unchanged inference.',
        restore_requirements=['Restore exact source-bound private model and saved persistent state; no zero-state substitution.',
                              'Restore actual warm Adam moments/counters and RNG, including pending accumulated gradients if present.',
                              'Freeze same factual race history and every forced suffix RNG; preserve causal token/label contract.'],
        fit_only_protocol=dict(anchors=['zero','factual winner return','local delivered-message surrogate','persistent-write-aware return predictor'],
            calibration='Use disjoint original FIT-only spans; no DEV/test to train or select the predictor. Freeze anchors/probabilities before fresh Bernoulli draws.',
            enumeration='One prespecified tiny production-model chunk with at most eight sampled terms; enumerate all masks and compare every raw-gradient mean/covariance against full same-history replay.',
            production='Prespecified 16-token chunk, original depth8/private architecture and optimizer; finite warm-Adam forks plus separate unused FIT-anchor predictions.',
            selection='Positive probability floor on every nonzero unresolved term; no pruning, observed-return stopping, or free oracle residual allocation.'),
        mandatory_contracts=['Every-parameter unbiased conditional mean in float64; record float32 error beside original thresholds.',
            'Bitwise factual logits, persistent state and factual-end RNG; all forced route-history identities.',
            'Anchor returns and inclusion probabilities detached; no surrogate gradient through predictor in main route auxiliary.',
            'All-mask moments include factual gradient and proper target/batch/accumulation normalization.',
            'Full and partially accumulated Adam recovery reproduce next raw gradients, moments, weights, cursor and activity counters.',
            'Clipped warm-Adam changes and disjoint FIT predictions compared with same-noise full replay, not inferred from unbiased gradients.'],
        cost_contract=['Trace candidate discovery, all-anchor inference/backward, predictor fitting, residual shadows, weighting, clip/Adam, and recovery verification separately.',
                       'Report whole-step GFLOPs, MFLOPs per presented target, inference work and wall/RSS; traffic/energy unknown unless measured.',
                       'Shared prefix/horizon reuse requires actual incremental cost and kernel runtime; fewer events alone do not establish speed.'],
        admission='Only after assigned AWS90M arms and an available guarded host slot. Unique one-job queue through run_safe.sh, source-frozen, bounded RSS/timeout and 8GiB available floor. No training queue admitted by this protocol file.')
    out=ROOT/'experiments/results/diagnostics/aws_residual_credit_protocol_20261003.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],checkpoint=result['checkpoint'],trained_targets=targets)))
