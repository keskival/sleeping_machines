"""Read-only matched-prefix diagnostics; never substitutes training scores for DEV."""
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments'),str(Path(__file__).parent)]
import torch
from aws_language_checkpoint_plasticity import analyze


def main():
    torch.set_num_threads(1)
    tags=[('private_replay','aws_language_winner_matrix_20261003T014100Z_private_replay_10m_s7'),
          ('private_teacher','aws_depth8_language_20261002T234100Z_private_pilot_s7'),
          ('shared_teacher','aws_depth8_language_20261002T234100Z_depth_pilot_s7')]
    rows=[];hashes=set();exposures={};updates={}
    for arm,tag in tags:
        prior_targets=0;prior_loss=0.
        for milestone in (1,2,3,4):
            path=ROOT/'experiments/results/aws_language_progress'/f'{tag}_milestone{milestone:03d}.pt'
            metadata=json.loads(path.with_suffix('.json').read_text())
            checkpoint=torch.load(path,weights_only=False)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['checkpoint_sha256']
            assert checkpoint['total_targets']==metadata['trained_targets']
            hashes.add(checkpoint['result']['fitting_data_sha256'])
            targets=metadata['trained_targets'];loss=metadata['cursor']['total_loss']
            assert targets>prior_targets and math.isfinite(loss)
            exposure=exposures.setdefault(milestone,targets);assert exposure==targets
            count=updates.setdefault(milestone,metadata['optimizer_updates']);assert count==metadata['optimizer_updates']
            plasticity=analyze(path);parameters=plasticity['rows']
            gates=[r['sigmoid_bias_only'] for r in parameters if 'sigmoid_bias_only' in r]
            rows.append(dict(arm=arm,milestone=milestone,checkpoint=str(path.relative_to(ROOT)),
                checkpoint_sha256=metadata['checkpoint_sha256'],targets=targets,optimizer_updates=count,
                interval_start_target=prior_targets,interval_end_target=targets,
                online_interval_bpc=(loss-prior_loss)/(targets-prior_targets)/math.log(2),
                cumulative_online_bpc=loss/targets/math.log(2),
                gate_output_coordinates=sum(r.get('weight',{}).get('count',0) for r in parameters),
                epsilon_dominated_coordinates=sum(r.get('epsilon_dominated_coordinates',0) for r in parameters),
                bias_only_sigmoid_range=[min(r['minimum'] for r in gates),max(r['maximum'] for r in gates)]))
            prior_targets,prior_loss=targets,loss
    assert len(hashes)==1,'Different fitting data'
    result=dict(status='completed',fitting_data_sha256=next(iter(hashes)),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,
        scope='Read-only saved equal-prefix checkpoints, identical fitting-data hash and Adam counts. Single-seed online predictions made during fitting, NOT DEV/test/generalization/completed10M quality or a statistically confirmed advantage. Initial DEV timing and implementation differ; RNG pairing not asserted. Gate bias and historical moments do not measure actual gate activity or causal usefulness of depth.')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
