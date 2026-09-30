"""Read-only experimental review, including adversarial causality probes.

Run only through queue/run_safe.sh. Completed means the audit executed, not
that every historical claim passed. Original results/checkpoints are immutable.
"""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import e62_charlm as count_lm
import e63_mixlm as old_word
import e79_race_mixer as race_lm
from e64_lm_baselines import LSTMLM, TfLM
from e52_thp import THP, encode, bank
from e120_shared_bench import BUILDERS, inputs, evaluate
from e123_synthetic_baselines import (EventTransformer, EventLSTM, CountedTransformer,
                                    batch, predict, evaluate as evaluate_dense)
from e120_shared_tasks import modular, recall
from e42_when import decision_points
from e139_fine_packet_model import load_marked, marked_packets
from e143_event_state_shd import event_batch
from e152_nuisance_state_shd import evaluate as evaluate_state
from sleeping_machines.causal_contexts import word_codes_before
from sleeping_machines.shared_event import SharedEventModel
from sleeping_machines.event_query import ObservedPrefix, pack_queries
from sleeping_machines.event_state import CoalescedEventStateEncoder, EventStateEncoder


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def word_probes():
    train = np.random.default_rng(171).integers(0, 27, 100000)
    train[::7] = 0
    stream = np.tile(np.array([1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 0]), 12)
    orders = [count_lm.Order(train, k) for k in range(4)]
    word = old_word.WordOrder(train)
    initial, sel = race_lm.expert_logp(orders, word, train, stream, 3, 256)
    cases = []
    for t in (6, 10, 15, 28, 65):
        changed = stream.copy()
        changed[t:] = np.random.default_rng(t).integers(1, 27, len(stream)-t)
        changed[t] = 1 if stream[t] == 0 else 0
        alternative, alt_sel = race_lm.expert_logp(orders, word, train, changed, 3, 256)
        # x[:t] has not changed; the entire distribution at t must be equal.
        cases.append(dict(position=t, target_original=int(stream[t]), target_changed=int(changed[t]),
            old_word_context_changes=bool(old_word.word_codes(stream)[t] != old_word.word_codes(changed)[t]),
            fixed_word_context_changes=bool(word_codes_before(stream)[t] != word_codes_before(changed)[t]),
            expert_max_abs_logp_changes=np.max(np.abs(initial[t]-alternative[t]), axis=1).tolist(),
            selector_changes=bool(sel[t] != alt_sel[t])))
    # Exhaustive short-stream future mutation, including spaces and early positions.
    rng = np.random.default_rng(172)
    stream = rng.integers(0, 4, 60)
    fixed = word_codes_before(stream)
    fixed_errors = 0
    for t in range(len(stream)):
        changed = stream.copy(); changed[t:] = rng.integers(0, 4, len(stream)-t)
        fixed_errors += int(fixed[t] != word_codes_before(changed)[t])
    return dict(disposition='withdraw E79 highlighted mixture scores; rebuild word counts and rerun',
                expert_order=['KT0', 'KT1', 'KT2', 'KT3', 'Witten-Bell', 'partial-word', 'copy'],
                cases=cases, fixed_context_future_mutation_failures=fixed_errors,
                legacy_current_target_space_reset=True)


@torch.no_grad()
def causal_baselines():
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    changed = x.clone(); changed[:, 7:] = torch.tensor([[26, 25, 24, 23, 22]])
    rows = {}
    for name, model in [('lstm', LSTMLM(16)), ('transformer', TfLM(16, 2, 16))]:
        model.eval()
        z, _ = model(x); alt, _ = model(changed)
        rows[name] = dict(max_prefix_logit_change=float((z[:, :7]-alt[:, :7]).abs().max()),
                          future_start=7, next_character_alignment='logit at input i predicts target i+1')
    T = np.arange(20, dtype=float)*.2
    Y = np.arange(20) % 4
    altered_T = T.copy(); altered_T[8:] += np.arange(1, 13)*.1
    altered_Y = Y.copy(); altered_Y[8:] = (Y[8:]+1)%4
    enc, alt = encode(T, Y, bank(0)), encode(altered_T, altered_Y, bank(0))
    model = THP(16, 20, enc[1].shape[1], 6).eval()
    a, b = model(enc[0][None], enc[1][None]), model(alt[0][None], alt[1][None])
    rows['market_thp'] = dict(max_prefix_intensity_change=float((a[:, :8]-b[:, :8]).abs().max()),
        max_prefix_feature_change=float((enc[1][:8]-alt[1][:8]).abs().max()),
        objective_gap_after_position_separate_from_features=True)
    # Count the original E64 evaluation positions exactly, without loading 100M chars.
    n, context = 1000000, 256
    starts = list(range(0, n-context-1, context//2))
    scored = np.concatenate([np.arange(s+1+(0 if i==0 else context//2), s+context+1)
                             for i, s in enumerate(starts)])
    rows['e64_scoring_population'] = dict(native_positions=n, lstm_positions=n-1,
        transformer_positions=len(scored), transformer_first=int(scored[0]),
        transformer_last=int(scored[-1]), transformer_unique=len(np.unique(scored)),
        missing_positions=n-len(scored), note='No duplicates, but first character and final tail are omitted; scores are not exactly the same target population.')
    return rows


def market_probes():
    t = np.arange(40, dtype=np.int64)*1000000
    p = np.exp(np.arange(40)*.0001)
    cutoff = 20
    altered = p.copy(); altered[cutoff+1:] *= np.exp(np.arange(19)*.3)
    a, b = decision_points(t, p), decision_points(t, altered)
    # Construct a decisive held-day perturbation of the legacy quantile.
    fitting = [np.linspace(1, 10, 100) for _ in range(5)]
    held = [np.linspace(1, 10, 100) for _ in range(2)]
    changed_held = [h*1000 for h in held]
    return dict(legacy_threshold_before=float(np.quantile(np.concatenate(fitting+held), .99)),
        legacy_threshold_after_held_only_change=float(np.quantile(np.concatenate(fitting+changed_held), .99)),
        fit_only_threshold_before=float(np.quantile(np.concatenate(fitting), .99)),
        fit_only_threshold_after_held_only_change=float(np.quantile(np.concatenate(fitting), .99)),
        price_decision_prefix_equal=bool(np.array_equal(a[a<=cutoff], b[b<=cutoff])),
        affected_sources=['e44_tpp.py', 'e47_worldrep.py', 'e48_event_world.py',
                          'e49_gru_offline.py', 'e52_thp.py', 'e57_regime.py'],
        affected_claim='Native market within .08-.19 nats of Transformer at ~1/3000 work',
        disposition='Quarantine historical held-day comparison; common E120 day25-only threshold is separate')


def metric_equal(measured, recorded):
    out = {name: dict(measured=measured[name], recorded=recorded[name],
                    absolute_error=abs(measured[name]-recorded[name]))
           for name in ('nll', 'accuracy', 'correct', 'n') if name in recorded and name in measured}
    if 'predictions' in recorded:
        out['predictions_equal'] = measured.get('predictions') == recorded['predictions']
    return out


@torch.no_grad()
def common_screens():
    rows = []
    for name in ('language', 'market', 'temporal', 'mnist', 'dvs', 'modular', 'recall'):
        source = Path('experiments/results/e120')/f'{name}_d8_20260929.json'
        if name == 'recall': source = source.with_name('recall_d2_20260929.json')
        if name == 'modular': source = Path('experiments/results/e124/modular_phase_only_s6_200.json')
        result = json.loads(source.read_text())
        if name == 'modular': task = modular(1473, 3440, 6)
        elif name == 'recall': task = recall(512, 256, 6)
        else: task = BUILDERS[name](result['args']['fit'], result['args']['dev'], 6)
        saved = torch.load(source.with_suffix('.pt'), map_location='cpu', weights_only=False)
        model = SharedEventModel(**saved['config']); model.load_state_dict(saved['state_dict']); model.eval()
        for r in task.fit+task.dev: r.prefix.validate()
        fitids, devids = {r.identity for r in task.fit}, {r.identity for r in task.dev}
        score = evaluate(model, task.dev, 16)
        original_inputs = inputs(task.dev[:4])
        target_changed = [dataclasses.replace(r, label=(r.label+1)%task.config['classes'],
             bucket=0 if r.bucket is not None else None,
             exposure=np.zeros_like(r.exposure) if r.exposure is not None else None)
             for r in task.dev[:4]]
        changed_inputs = inputs(target_changed)
        target_inputs_equal = all(torch.equal(v, changed_inputs[k]) if isinstance(v, torch.Tensor)
                                  else v == changed_inputs[k] for k, v in original_inputs.items())
        individual = model(**inputs(task.dev[:1]))[0]
        together = model(**inputs(task.dev[:4]))[0][:1]
        row = dict(task=name, result=str(source), fit_dev_id_overlap=len(fitids & devids),
            declared_prefixes_valid=True, target_mutation_inputs_equal=target_inputs_equal,
            batch_other_query_logit_effect=float((individual-together).abs().max()),
            score_reproduction=metric_equal(score, result['final']['dev']),
            parameter_count=sum(p.numel() for p in model.parameters()), protocol=task.protocol)
        if name in ('language', 'market', 'temporal', 'mnist', 'dvs', 'modular', 'recall'):
            base = Path('experiments/results/e123')/(f'{name}_tf_counts_d32_l2_s6.json'
                        if name=='dvs' else f'{name}_tf_d32_l2_s6.json')
            control = json.loads(base.read_text())
            cp = torch.load(base.with_suffix('.pt'), map_location='cpu', weights_only=False)
            net = (CountedTransformer(task.config['bands'], task.config['classes'],32,2) if name=='dvs'
                   else EventTransformer(task.config['bands'], task.config['classes'],32,2,1.2))
            net.load_state_dict(cp['state_dict']); net.eval()
            ref = evaluate_dense(net, task.dev, 16)
            b,t,mask,_ = batch(task.dev[:4])
            joint = predict(net,b,t,mask,task.dev[:4])[:1]
            b,t,mask,_ = batch(task.dev[:1])
            alone = predict(net,b,t,mask,task.dev[:1])
            row.update(transformer_score_reproduction=metric_equal(ref,control['final']['dev']),
                transformer_batch_other_query_effect=float((joint-alone).abs().max()),
                identical_dev_ids=control['dev_ids']==[r.identity for r in task.dev],
                baseline_batch_size=control['args']['bs'], common_batch_size=result['args'].get('bs'),
                neural_passes_common=result['args']['epochs'], neural_passes_reference=control['args']['epochs'],
                extra_evidence_supplied_to_reference=False)
        if name in ('recall', 'modular'):
            base = Path('experiments/results/e123')/f'{name}_lstm_d32_l2_s6.json'
            control=json.loads(base.read_text()); cp=torch.load(base.with_suffix('.pt'),weights_only=False,map_location='cpu')
            net=EventLSTM(task.config['bands'],task.config['classes'],32,2)
            net.load_state_dict(cp['state_dict']); net.eval()
            row['lstm_score_reproduction']=metric_equal(evaluate_dense(net,task.dev,16),control['final']['dev'])
        print(json.dumps({'audited_task':name, 'nll':score['nll']}),flush=True)
        rows.append(row)
    return rows


@torch.no_grad()
def event_causality():
    torch.manual_seed(171)
    b=torch.arange(12)%8; t=torch.arange(12,dtype=torch.float32)*.03
    model=SharedEventModel(bands=8,groups=1,dim=8,depth=4,classes=3,memory_backend='linear').eval()
    args=dict(b=b,t=t,c=torch.ones(12),ids=torch.zeros(12,dtype=torch.long),size=1,trace=True)
    _,_,_,trace=model(**args)
    args['b']=b.clone();args['b'][7:]=(args['b'][7:]+3)%8
    _,_,_,altered=model(**args)
    packet_errors=[float((a['out'][:7]-b['out'][:7]).abs().max()) for a,b in zip(trace,altered)]
    # Fixed nonempty closures at >= every source timestamp.
    times=np.array([.001,.009,.010,.019,.051])
    units=np.array([0,699,350,7,6])
    packed=marked_packets(times,units)
    lag=packed[1][packed[3]]-packed[5]
    encoder=EventStateEncoder(8,width=8,modes=4,depth=6,classes=3).eval()
    x=torch.randn(12,8); changed=x.clone();changed[7:]+=10
    a=encoder(x,t,torch.zeros(12,dtype=torch.long),torch.ones(12),1)[1]
    c=encoder(changed,t,torch.zeros(12,dtype=torch.long),torch.ones(12),1)[1]
    return dict(shared_layer_prefix_payload_changes=packet_errors,
        state_encoder_prefix_payload_change=float((a[:7]-c[:7]).abs().max()),
        packet_min_closure_minus_raw_time=float(lag.min()), packet_release_at_end_of_fixed_bin=True,
        end_readout_is_completed_prefix_only=True,
        warning='End-of-query pooled classification is not a tested real-time early-confidence policy. Classification Transformers use bidirectional attention within the already observed prefix.')


@torch.no_grad()
def speech_review():
    source=Path('experiments/results/e165/selected_prefix_20260930.json')
    result=json.loads(source.read_text()); cp=torch.load(source.with_suffix('.pt'),weights_only=False,map_location='cpu')
    cfg=cp['encoder_config']
    encoder=CoalescedEventStateEncoder(sources=720,width=cfg['width'],modes=cfg['modes'],depth=cp['actual_depth'])
    encoder.load_state_dict(cp['encoder_state_dict']);encoder.eval()
    fit=load_marked(6144,'fit_spk',6);held=load_marked(512,'val_spk',7)
    metric=evaluate_state(encoder,held,4,.01)
    return dict(fit_dev_absolute_id_overlap=len({r[8] for r in fit}&{r[8] for r in held}),
        fit_n=len(fit),dev_n=len(held),metric=metric,
        expected_correct=408,expected_nll=.62516063,
        official_test_read=False,selection='Private development used repeatedly; selected prefix/full choice uses this sample. Reused 657 audit is not a fresh independent selection set.',
        prefix_time_basis='Original recorded onset and fixed bin closures, not normalization by future duration')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args()
    out=Path('experiments/results/e171')/(a.tag+'.json');out.parent.mkdir(exist_ok=True)
    if out.exists() or Path(a.tag).name!=a.tag:raise ValueError('Unique output required')
    torch.set_num_threads(1);torch.manual_seed(171);start=time.perf_counter()
    result=dict(status='running',word_context=word_probes(),baseline_causality=causal_baselines(),
        market=market_probes(),event_causality=event_causality(),
        hardware=dict(platform=platform.platform(),torch=torch.__version__,threads=1,device='cpu'))
    out.write_text(json.dumps(result,indent=2)+'\n')
    result['common_screens']=common_screens()
    result['speech']=speech_review()
    sources=[Path(__file__),*Path('sleeping_machines').glob('*.py'),
             *[Path('experiments')/f for f in ('e62_charlm.py','e63_mixlm.py','e79_race_mixer.py',
                'e64_lm_baselines.py','e120_shared_tasks.py','e123_synthetic_baselines.py',
                'e36_transformer.py','e52_thp.py','e57_regime.py','e48_event_world.py')]]
    result.update(status='completed',source_sha256={str(p):digest(p) for p in sources},
        wall_s=time.perf_counter()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Causality counterexamples; all highlighted consolidated development scores reconstructed; speech selected-prefix score; native synthetic source review documented separately. No repaired large-run score invented.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'completed':str(out),'wall_s':result['wall_s']}),flush=True)


if __name__=='__main__':main()
