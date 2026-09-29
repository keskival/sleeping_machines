"""Full-depth value learning under computed fixed or coupled key schedules."""
import hashlib
from pathlib import Path
import torch
from sleeping_machines.shared_event import SharedEventModel

SOURCE = Path('experiments/results/e122/d8_n4096_invariance_continue_s6_e2.pt')
CACHE = Path('experiments/results/e122/d8_n4096_shared_values_s6.json')
CONTEXT = (3, 7)


def build(key_mode):
    if key_mode not in ('fixed','coupled'):
        raise ValueError('Unknown key mode')
    torch.manual_seed(6)
    saved=torch.load(SOURCE,weights_only=False,map_location='cpu')
    model=SharedEventModel(depth=8,memory_backend='linear',global_context_layers=CONTEXT,cf_credit=False)
    missing,unexpected=model.load_state_dict(saved['state_dict'],strict=False)
    assert not unexpected and set(missing)=={f'layers.{j}.bridge_{kind}' for j in CONTEXT for kind in ('value','route')}
    if key_mode=='fixed':model.freeze_keys_from_values()
    train_names=[]
    for name,param in model.named_parameters():
        is_policy=name.startswith('key_') or any(name.endswith('.'+n) for n in ('route','route_bias','bridge_route'))
        param.requires_grad_(not is_policy)
        if param.requires_grad:train_names.append(name)
    assert all(any(n.startswith(f'layers.{j}.') for n in train_names) for j in range(8))
    assert not any(n.startswith('key_') for n in train_names)
    config=dict(depth=8,memory_backend='linear',global_context_layers=CONTEXT,
                cf_credit=False,separate_keys=key_mode=='fixed')
    return model,train_names,config


def hashes(driver):
    paths=[Path(driver),Path(__file__),Path('experiments/e122_shd_continuation.py'),
           Path('experiments/e117_serial_event_shd.py'),Path('experiments/e118_race_carrier_shd.py'),
           *Path('sleeping_machines').glob('*.py')]
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
