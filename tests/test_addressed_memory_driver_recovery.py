import json
import sys
import torch
import pytest

sys.path.insert(0,'experiments')
import addressed_memory_language_benchmark as driver


@pytest.mark.parametrize('kind',['native','addressed','late'])
def test_whole_driver_interruption_recovers_cursor_state_rng_work_and_next_updates(tmp_path,monkeypatch,kind):
    original=driver.ROOT
    (tmp_path/'sleeping_machines').symlink_to(original/'sleeping_machines',target_is_directory=True)
    (tmp_path/'experiments').mkdir()
    for source in (original/'experiments').glob('*.py'):
        (tmp_path/'experiments'/source.name).symlink_to(source)
    monkeypatch.setattr(driver,'ROOT',tmp_path)
    def run(tag,*extra):
        monkeypatch.setattr(sys,'argv',['driver','--tag',tag,'--model',kind,'--fit','160','--dev','65',
            '--epochs','2','--payload','2','--depth','1','--chunk','16','--update-targets','64',
            '--warmup-targets','64',*extra])
        driver.main()
    run('continuous')
    run('recovered','--stop-after-updates','1')
    directory=tmp_path/'experiments/results/addressed_memory_language'
    assert not (directory/'recovered.json').exists()
    paused=torch.load(directory/'recovered.progress.pt',weights_only=False)
    assert paused['updates']==1 and paused['cursor']['next_target']==64
    assert paused['stream_state'].events==64
    run('recovered','--resume')
    left=torch.load(directory/'continuous.progress.pt',weights_only=False)
    right=torch.load(directory/'recovered.progress.pt',weights_only=False)
    for key in ('model','initial','best_state'):
        for name,value in left[key].items():torch.testing.assert_close(value,right[key][name],rtol=0,atol=0)
    assert left['optimizer']['param_groups']==right['optimizer']['param_groups']
    for key,value in left['optimizer']['state'].items():
        for name,tensor in value.items():torch.testing.assert_close(tensor,right['optimizer']['state'][key][name],rtol=0,atol=0)
    torch.testing.assert_close(left['torch_rng'],right['torch_rng'],rtol=0,atol=0)
    for key in ('final','work','parameter_delta','selected_epoch','fitting_data_sha256','development_data_sha256'):
        assert left['result'][key]==right['result'][key],key
    assert left['total_targets']==right['total_targets']==318
    assert left['updates']==right['updates']==6
    assert left['result']['final']['frozen_fit_replay_dev']['replay_tokens']==160
    assert left['result']['final']['frozen_fit_replay_dev']['n']==64


def test_replay_boundary_does_not_write_an_invented_fit_to_dev_transition():
    from sleeping_machines.late_projected_context_memory import LateProjectedContextModel
    torch.manual_seed(6)
    model=LateProjectedContextModel(payload=2,depth=1,heads=2,pool=2,order=1,buckets=64)
    prefix=torch.tensor([1,2,3,4]);dev=torch.tensor([5,6,7])
    row=driver.evaluate(model,dev,1,prefix=prefix)
    # Two dev observations make only one dev outcome write; the first observed
    # dev symbol must not be stored as what followed the last fitting context.
    assert row['activity']['context_reads']==2
    assert row['activity']['context_writes']==1
    assert row['replay_tokens']==4 and row['n']==2
