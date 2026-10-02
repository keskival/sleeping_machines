"""Frozen inference fusion and prefix-information probes after completed fits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace
import torch
from torch.nn import functional as F
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from addressed_memory_language_benchmark import make_model
from language_learning_audit import fingerprint,piece
from sleeping_machines.fused_context_reader import compile_reader
from sleeping_machines.context_addressed_memory import ContextAddressedNativeModel
from sleeping_machines.count_carrying_language import eval_stream_counts
from race_language_screen import capture
from paired_parity_protocol import examples
from e120_shared_tasks import text_slice
from count_reference_language import score as count_score


def load(row):
    path=ROOT/row['result'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['result_sha256']
    r=json.loads(path.read_text());assert r['status']=='completed'
    checkpoint=path.with_suffix('.progress.pt');saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert saved['result']['status']=='completed' and saved['result']['final']==r['final']
    for name,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
    model=make_model(SimpleNamespace(**r['args']));model.load_state_dict(saved['model']);model.eval()
    return model,r,hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def fusion_audit(model,dev):
    before=fingerprint(model);started=time.perf_counter();fused=compile_reader(model);compile_wall=time.perf_counter()-started
    d=model.total_payload
    fold=capture(lambda:model.memory_read_map.weight[:,:d]@model.memory_write_map.weight)
    old=ContextAddressedNativeModel(payload=model.payload,depth=model.depth,heads=model.heads,pool=model.pool,
        order=model.order,buckets=model.buckets);old.load_state_dict(model.state_dict());old.eval()
    outputs=[]
    with torch.no_grad(),torch.random.fork_rng():
        for m in (model,fused,old):
            z,_=piece(m,dev,0,256,True);outputs.append(z)
    for z in outputs[1:]:torch.testing.assert_close(z,outputs[0],atol=1e-4,rtol=1e-4)
    traces=[];occupied=[]
    for m in (model,fused):
        with torch.no_grad(),torch.random.fork_rng():
            _,state=piece(m,dev,0,128,True);count=[0]
            hook=None
            if m is model:
                hook=m.memory_write_map.register_forward_hook(lambda *_:count.__setitem__(0,count[0]+1))
            def sample():
                z,_=piece(m,dev,128,144,True,state)
                F.cross_entropy(z,dev[129:145],reduction='sum')
            traces.append(capture(sample));occupied.append(count[0])
            if hook is not None:hook.remove()
    unit=lambda r:r['arithmetic_flops']+r['special_function_evaluations']
    savings=unit(traces[0])-unit(traces[1]);expected=2*d*d*occupied[0]
    assert savings==expected,(savings,expected)
    assert fingerprint(model)==before
    return dict(frozen_text_input_tokens=256,model_fingerprint=before,weights_preserved=True,
        max_fused_logit_difference=float((outputs[1]-outputs[0]).abs().max()),
        max_original_projection_placement_logit_difference=float((outputs[2]-outputs[0]).abs().max()),
        compile_wall_s=compile_wall,fold_arithmetic_trace=fold,
        fitting_parameters=sum(p.numel() for p in model.parameters()),
        compiled_inference_parameters=sum(p.numel() for p in fused.parameters()),
        removed_parameters=d*d,inference_targets=16,inference_warm_tokens=128,
        occupied_reads=occupied[0],unfused_cpu_unit_special_flops_per_target=unit(traces[0])/16,
        fused_cpu_unit_special_flops_per_target=unit(traces[1])/16,
        measured_sample_savings=savings,expected_projection_only_savings=expected,
        break_even_occupied_reads=unit(fold)/(2*d*d),traces=traces,
        scope='Frozen-reader algebra/coupled 256-token sample, not full development quality or latency/energy. '
              'Fold trace charges matrix product; compiler initialization, copying and RNG outside FLOPs. '
              'Training is explicitly refused; compile is not a fitting architecture substitution.')


def parity_audit(model,fit):
    evidence=[];forward_tokens=0
    with torch.no_grad(),torch.random.fork_rng():
        for gap in (8,32,64):
            pairs=examples(gap=gap,suffixes=4);logits=[];features=[];count_identity=True
            for row in pairs:
                inputs=torch.tensor(row['inputs']);z,state=piece(model,inputs,0,len(inputs),True)
                logits.append(z[-1].double().log_softmax(-1));features.append(state.contexts[0][0].double())
                forward_tokens+=len(inputs)
            sensitivity=[];feature_difference=[]
            for group in range(4):
                rows=pairs[group*4:group*4+4];vectors=[]
                for row in rows:
                    stream=[*row['inputs'],row['target']]
                    vectors.append(eval_stream_counts(fit,stream,8)[:,len(row['inputs'])-1])
                for c in vectors[1:]:
                    assert __import__('numpy').array_equal(c,vectors[0])
                a,b=logits[group*4],logits[group*4+1]
                sensitivity.append(float((a.exp()*(a-b)).sum()))
                feature_difference.append(float((features[group*4]-features[group*4+1]).norm()))
            evidence.append(dict(gap=gap,pairs=4,examples=16,query_counts_orders_1_to_8_identical=count_identity,
                balanced_local_predictor_logloss_lower_bound_bits=1.,full_prefix_oracle_logloss_bits=0.,
                mean_paired_query_kl_nats=sum(sensitivity)/len(sensitivity),
                mean_top_context_feature_difference_norm=sum(feature_difference)/len(feature_difference)))
    return dict(rows=evidence,forward_input_tokens=forward_tokens,
        scope='Frozen text checkpoints never trained on parity: prefix sensitivity/retention only, no parity-learning score. '
              'Bound applies to the balanced query suffix/count-vector restriction, not arbitrary recurrent count processing.')


def head_features(model,tokens):
    features=[];handle=model.head.register_forward_pre_hook(lambda _m,args:features.append(args[0].detach()))
    with torch.no_grad(),torch.random.fork_rng():
        torch.manual_seed(314159);state=model.new_state()
        for begin in range(0,len(tokens)-1,16):
            end=min(begin+16,len(tokens)-1);_,state=model.forward_chunk(tokens[begin:end],state)
    handle.remove();x=torch.stack(features).double();targets=tokens[1:]
    with torch.no_grad():
        logits=F.linear(x,model.head.weight.double(),model.head.bias.double());p=logits.softmax(-1)
        error=p-F.one_hot(targets,model.vocabulary)
        gweight=error.T@x/len(targets);gbias=error.mean(0)
        centered=x-x.mean(0);covariance=centered.T@centered/len(targets)
        spectrum=torch.linalg.eigvalsh(covariance).clamp_min(0)
        participation=float(spectrum.sum().square()/spectrum.square().sum().clamp_min(1e-30))
    return dict(targets=len(targets),frozen_bpc=float(F.cross_entropy(logits,targets))/math.log(2),
        mean_head_weight_gradient_norm=float(gweight.norm()),mean_head_bias_gradient_norm=float(gbias.norm()),
        feature_covariance_participation_rank=participation,feature_covariance_trace=float(spectrum.sum()),
        feature_covariance_spectrum=spectrum.tolist(),
        scope='Exact fixed-feature linear-head loss derivatives/geometry; no head fitting or optimizer step. '
              'Nonstationarity under finite joint fitting and covariance rank alone do not establish a bug or semantic depth.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    out=ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if Path(a.tag).name!=a.tag or out.exists():raise ValueError('Unused tag required')
    started=time.perf_counter();torch.set_num_threads(1);plan=json.loads((ROOT/a.plan).read_text())
    status=json.loads((ROOT/a.plan).with_suffix('.status.json').read_text());assert status['status']=='completed'
    analysis_path=next(j['result'] for j in plan['jobs'] if j['stage']=='analysis')
    analysis=json.loads((ROOT/analysis_path).read_text());assert analysis['status']=='completed'
    tests=['tests/test_fused_context_reader.py','tests/test_paired_parity_protocol.py','tests/test_joint_feature_credit.py']
    assert pytest.main(['-q',*tests])==0,'Fusion or conditional-information contract failed'
    dev=torch.tensor(text_slice(90000000,2048));fit=__import__('numpy').array(text_slice(0,1024))
    count_controls=[]
    for order in (1,2,3,4):
        for method,adaptive in (('kn',False),('wb',False),('wb',True),('ad',False),('ad',True)):
            mark=time.perf_counter();bpc,lookups=count_score(fit,dev.numpy(),order,method,adaptive)
            count_controls.append(dict(order=order,method=method,adaptive=adaptive,bpc=bpc,
                fitting_characters=1024,passes=1,development_targets=2047,
                represented_fit_count_increments=sum(1024-k for k in range(order+1)),
                dev_table_lookups=lookups,dev_count_increments=lookups if adaptive else 0,
                wall_s=time.perf_counter()-mark))
    rows=[]
    for row in analysis['common_unit_ledger']:
        model,r,sha=load(row);before=fingerprint(model)
        entry=dict(arm=row['arm'],result=row['result'],result_sha256=row['result_sha256'],checkpoint_sha256=sha,
            prefix_information=parity_audit(model,fit),
            head_features=dict(fitting=head_features(model,torch.tensor(fit)),development=head_features(model,dev)))
        assert abs(entry['head_features']['development']['frozen_bpc']-row['cold_bpc'])<1e-5
        if r['args']['model']=='late':entry['fusion']=fusion_audit(model,dev)
        assert fingerprint(model)==before
        rows.append(entry);print(json.dumps(dict(completed=row['arm'])),flush=True)
    names=['experiments/value_projection_frozen_audit.py','sleeping_machines/fused_context_reader.py',
        'experiments/paired_parity_protocol.py','experiments/language_learning_audit.py',
        'experiments/count_reference_language.py','experiments/e120_shared_tasks.py',*tests]
    result=dict(status='completed',args=vars(a),models=rows,count_controls=count_controls,
        count_control_scope='All orders/methods separately reported on exact1K/2047-target windows, fixed discounts. '
            'Fit-prefilled count tables and optional dev adaptation differ from cold learned-model state; '
            'warm learned replay also has prior fit observations. One count pass versus four gradient passes. '
            'Integer table construction/lookup work and wall time separate, not converted to neural FLOPs; '
            'equal bpc is calibration, not proof of equal features or iso-FLOP superiority.',
        optimizer_steps=0,weights_preserved=True,
        official_test_read=False,analysis_result=analysis_path,
        analysis_sha256=hashlib.sha256((ROOT/analysis_path).read_bytes()).hexdigest(),
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},
        wall_s=time.perf_counter()-started,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        hardware=dict(host=__import__('os').uname().nodename,device='cpu',threads=1,torch=torch.__version__))
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=a.tag,wall_s=result['wall_s'])),flush=True)


if __name__=='__main__':main()
