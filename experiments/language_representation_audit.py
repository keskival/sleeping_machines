"""Frozen development interventions on completed learned language checkpoints.

These measure fitted dependence on content and history, not the achievable
quality of retrained ablated architectures. Run only through a guarded queue.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
from e120_shared_tasks import text_slice
from parallel_event_language import evaluate
from sleeping_machines.parallel_stream_language import ParallelEventLanguageModel
from sleeping_machines.selective_stream_language import SelectiveEventLanguageModel


@torch.no_grad()
def reset_history_score(model, tokens):
    model.eval(); total = 0.
    for i in range(len(tokens)-1):
        state = model.new_state(); state.position = i
        logits, _ = model.forward_chunk(tokens[i:i+1], state)
        total += float(F.cross_entropy(logits, tokens[i+1:i+2], reduction='sum'))
    return dict(n=len(tokens)-1,bpc=total/(len(tokens)-1)/math.log(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--results', nargs='+', required=True)
    args = parser.parse_args()
    output = ROOT/'experiments/results/parallel_language'/f'{args.tag}.json'
    if Path(args.tag).name != args.tag or output.exists():
        raise ValueError('A unique plain tag is required')
    started = time.perf_counter(); torch.set_num_threads(1)
    rows = []
    for name in args.results:
        path = ROOT/name; result = json.loads(path.read_text())
        if result['status'] != 'completed' or result['protocol']['official_test_read']:
            raise ValueError('Completed development-only models required')
        model_sources = [source for source in result['source_sha256'] if source.startswith('sleeping_machines/')]
        for source in model_sources:
            if hashlib.sha256((ROOT/source).read_bytes()).hexdigest() != result['source_sha256'][source]:
                raise ValueError('Model source differs from fitted checkpoint: '+source)
        checkpoint = path.with_suffix('.progress.pt')
        saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['result']['status'] != 'completed' or saved['result']['final'] != result['final']:
            raise ValueError('Checkpoint/result mismatch')
        settings = result['args']
        model_class = SelectiveEventLanguageModel if settings.get('content_memory') else ParallelEventLanguageModel
        model = model_class(width=settings['width'],modes=settings['modes'],depth=settings['depth'])
        model.load_state_dict(saved['model']); model.eval()
        start,end = result['protocol']['development']; tokens = torch.tensor(text_slice(start,end-start))
        reference = evaluate(model,tokens,settings['chunk'])
        if abs(reference['bpc']-result['final']['dev']['bpc']) > 1e-6:
            raise ValueError('Frozen score does not reproduce result')
        interventions = dict(full=reference,reset_history_every_token=reset_history_score(model,tokens))
        probe = copy.deepcopy(model)
        with torch.no_grad():probe.embedding.weight.zero_()
        interventions['zero_incoming_embeddings'] = evaluate(probe,tokens,settings['chunk'])
        probe = copy.deepcopy(model)
        for layer in probe.layers:layer.gain=0.
        interventions['remove_all_memory_corrections'] = evaluate(probe,tokens,settings['chunk'])
        controls = None
        if settings.get('content_memory'):
            observations = [[] for _ in model.layers]
            original_controls = model.controls
            index = {id(layer):i for i,layer in enumerate(model.layers)}
            def tracked(layer,x):
                forget,write = original_controls(layer,x)
                observations[index[id(layer)]].append(torch.stack((forget,write),-1).detach())
                return forget,write
            model.controls = tracked
            evaluate(model,tokens,settings['chunk'])
            controls = []
            for i,values in enumerate(observations):
                values = torch.cat(values)
                controls.append(dict(layer=i+1,forget_quantiles=[float(v) for v in values[:,0].quantile(torch.tensor([.1,.5,.9]))],
                                     write_quantiles=[float(v) for v in values[:,1].quantile(torch.tensor([.1,.5,.9]))]))
        row = dict(result=name,result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   parameters=result['parameters'],args=settings,
                   interventions={key:dict(n=value['n'],bpc=value['bpc'],
                       delta_bpc=value['bpc']-reference['bpc']) for key,value in interventions.items()},
                   content_memory_controls=controls)
        rows.append(row)
        print(json.dumps(dict(model=name,interventions=row['interventions'])),flush=True)
    sources=[Path(__file__),ROOT/'experiments/parallel_event_language.py',ROOT/'experiments/e120_shared_tasks.py',
             ROOT/'sleeping_machines/parallel_stream_language.py',ROOT/'sleeping_machines/selective_stream_language.py',
             ROOT/'sleeping_machines/stream_language.py',ROOT/'sleeping_machines/event_state.py',ROOT/'sleeping_machines/event_memory.py']
    result=dict(status='completed',args=vars(args),rows=rows,wall_s=time.perf_counter()-started,
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                protocol='Same completed selected checkpoints and identical cold development targets. No fitting, parameter selection or official test. Reset-history retains the current character and removes every earlier modal state.',
                scope='Frozen fitted-dependence interventions, not retrained model comparisons, energy or quality superiority.',
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
