"""Read-only throughput forecast from immutable checkpoints, not model results."""
import datetime
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    rows=[]
    tags=['aws_depth8_language_20261002T234100Z_private_pilot_s7',
          'aws_depth8_language_20261002T234100Z_depth_pilot_s7',
          'aws_language_winner_matrix_20261003T014100Z_private_replay_10m_s7']
    for tag in tags:
        files=sorted((ROOT/'experiments/results/aws_language_progress').glob(tag+'_milestone*.json'))
        first,last=(json.loads(p.read_text()) for p in (files[0],files[-1]))
        targets=last['trained_targets']-first['trained_targets']
        wall=last['cumulative_wall_s']-first['cumulative_wall_s']
        assert targets>0 and wall>0
        rate=targets/wall
        rows.append(dict(tag=tag,first_checkpoint=str(files[0].relative_to(ROOT)),
            latest_checkpoint=str(files[-1].relative_to(ROOT)),saved_targets=last['trained_targets'],
            measured_checkpoint_interval_targets_per_s=rate,
            projected_remaining_training_hours=(10_000_000-1-last['trained_targets'])/rate/3600))
    matrix=json.loads((ROOT/'experiments/queue/aws_language_winner_matrix_20261003T014100Z/manifest.json').read_text())
    current={r['tag'] for r in rows}
    pending=[j['tag'] for j in matrix['jobs'] if j['stage']=='pilot' and j['tag'] not in current]
    print(json.dumps(dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),active=rows,
        additional_pending_10m_fits=pending,
        projected_minimum_host_reservation_delay_hours=max(r['projected_remaining_training_hours'] for r in rows),
        scope='Operational projection at saved checkpoints, assuming measured throughput persists. Final DEV, failures/recovery, pending fits and admission overhead excluded; full90M queue delay can be longer. No projected quality, speedup or completed benchmark claim. Read-only metadata; no model/data execution.'),indent=2))


if __name__=='__main__':main()
