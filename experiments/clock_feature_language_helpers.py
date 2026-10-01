"""Native receiver language protocol, numerical contracts and work identity."""
import copy
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

from integrated_language_protocol import learn_chunk
from clock_feature_event_contracts import contracts as event_contracts
from sparse_language_contracts import hashes
from parallel_head_race_language_screen import architecture_forward
from sleeping_machines.clock_feature_event_heads import ClockFeatureLanguageModel as NativeStreamLanguageModel
from native_language_helpers import source_hashes as parent_source_hashes

ROOT=Path(__file__).resolve().parents[1]


def source_hashes():
    names=('experiments/clock_feature_language_helpers.py','experiments/clock_feature_event_contracts.py',
        'sleeping_machines/clock_feature_event_heads.py','sleeping_machines/clock_feature_races.py',
        'experiments/native_event_tasks.py','sleeping_machines/addressed_event_heads.py',
        'sleeping_machines/native_stream_language.py','sleeping_machines/parallel_head_race_language.py',
        'experiments/parallel_head_race_language_screen.py','experiments/integrated_language_protocol.py')
    return {**parent_source_hashes(),**{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}


def activity(state,model):
    return dict(receiver_scores=state.candidate_scores,receiver_updates=state.selected_updates,
        receiver_teacher_values=state.counterfactual_values,race_count=state.selected_updates+state.extra_clock_races,
        scored_clock_rates=state.candidate_scores+state.extra_rate_settings,
        extra_clock_races=state.extra_clock_races,extra_clock_rate_settings=state.extra_rate_settings,
        receiver_persistent_tensor_bytes=state.storage()['persistent_tensor_bytes'],
        kv_queries=0,kv_scores=0,kv_delivered_values=0,kv_teacher_values=0,
        kv_stored_entries=0,kv_raw_key_value_bytes=0,kv_winner_age_max=0)


@torch.no_grad()
def evaluate(model,tokens,chunk):
    with torch.random.fork_rng():
        torch.manual_seed(314159);model.eval();state=model.new_state();total=0.
        for start in range(0,len(tokens)-1,chunk):
            end=min(start+chunk,len(tokens)-1)
            z,state=model.forward_chunk(tokens[start:end],state)
            total+=float(F.cross_entropy(z,tokens[start+1:end+1],reduction='sum'))
        return dict(n=len(tokens)-1,bpc=total/(len(tokens)-1)/0.6931471805599453,
                    activity=activity(state,model),persistent_state=state.storage())


def contracts(payload,depth,pool,matching,recent,heads,clock_features=4,clock_readout=True,clock_allocation='uniform'):
    if matching or recent:raise ValueError('No per-position retrieval candidates')
    checks=event_contracts(1,payload,depth,pool,heads,clock_features,clock_readout,clock_allocation)
    with torch.random.fork_rng():
        torch.manual_seed(947);model=NativeStreamLanguageModel(payload,depth,pool,heads,clock_features=clock_features,clock_readout=clock_readout,clock_allocation=clock_allocation)
        tokens=torch.tensor([1,2,1,3,1,2,4,1])
        rng=torch.get_rng_state()
        def forward(training,seq):
            torch.set_rng_state(rng);model.train(training)
            with torch.no_grad():return model.forward_chunk(seq)[0]
        z=forward(False,tokens);taught=forward(True,tokens)
        torch.testing.assert_close(z,taught,rtol=0,atol=0)
        future=tokens.clone();future[4:]=(future[4:]+7)%27
        other=forward(False,future)
        torch.testing.assert_close(z[:4],other[:4],rtol=0,atol=0)
        opt=torch.optim.Adam(model.parameters(),lr=.002)
        _,state,_=learn_chunk(model,opt,tokens[:4],tokens[1:5],model.new_state())
        restored=copy.deepcopy(model);other=torch.optim.Adam(restored.parameters(),lr=.002)
        other.load_state_dict(copy.deepcopy(opt.state_dict()));saved=copy.deepcopy(state)
        rng=torch.get_rng_state()
        loss,_,p=learn_chunk(model,opt,tokens[4:7],tokens[5:8],state)
        torch.set_rng_state(rng)
        recovered,_,q=learn_chunk(restored,other,tokens[4:7],tokens[5:8],saved)
        assert loss==recovered;torch.testing.assert_close(p,q,rtol=0,atol=0)
        for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
        checks.update(token_adapter_causal=True,token_teacher_equals_inference=True,
            exact_live_stream_next_update_recovery=True,shared_receivers_across_token_content=True)
    return checks
