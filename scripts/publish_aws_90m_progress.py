"""Archive and publish 90M optimizer checkpoints at 10M presentation milestones."""
import argparse,datetime,fcntl,hashlib,json,os,signal,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import publish_aws_language_progress as P


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);args=parser.parse_args()
    os.chdir(ROOT)
    if P.git('branch','--show-current')!='main':raise RuntimeError('Main required')
    torch.set_num_threads(1);manifest=Path(args.manifest);plan=json.loads(manifest.read_text())
    directory=ROOT/'experiments/results/aws_language_90m_progress';directory.mkdir(exist_ok=True)
    status=manifest.parent/'90m_progress_publisher.status.json'
    state=dict(status='monitoring',published=[])
    def save():
        tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status)
    seen={};save()
    while True:
        lifecycle=json.loads((manifest.parent/'worker_recovery.status.json').read_text())
        if lifecycle['status'] in ('completed','needs_review'):
            state['status']='stopped_'+lifecycle['status'];save();return
        for job in plan['admission_jobs']:
            if job['stage']!='fit':continue
            result=Path(job['result']);checkpoint=result.parent/'checkpoints'/(job['tag']+'.pt')
            if result.exists() or not checkpoint.exists():continue
            stamp=checkpoint.stat().st_mtime_ns
            if seen.get(job['tag'])==stamp:continue
            seen[job['tag']]=stamp
            saved=torch.load(checkpoint,weights_only=False)
            assert saved['args']['tag']==job['tag'] and saved['args']['fit']==90_000_000
            assert saved['seen']==saved['window']*saved['args']['segment']*saved['args']['lanes']
            milestone=saved['seen']//10_000_000
            if milestone<1:continue
            archive=directory/f"{job['tag']}_milestone{milestone:02d}.pt"
            metadata=archive.with_suffix('.json')
            if archive.exists():continue
            for name,digest in plan['source_sha256'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
            temporary=archive.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(archive)
            record=dict(status='checkpoint',args=saved['args'],presented_targets=saved['seen'],
                optimizer_windows=saved['window'],source_sha256=plan['source_sha256'],
                publisher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                checkpoint=str(archive.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                cumulative_wall_s=saved['wall_s'],captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='Partial90M training/recovery checkpoint with model/Adam/schedule/cursor/NumPy+torch RNG. Random segments; presentations are not unique coverage. Not finalDEV/test/comparable-quality result. Immutable milestone archive; originalcheckpoint untouched.')
            metadata.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            files=[str(archive.relative_to(ROOT)),str(metadata.relative_to(ROOT))]
            with open('/tmp/aws-language-publication.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX);pid=P.coordinator(args.manifest);os.kill(pid,signal.SIGSTOP)
                try:
                    for _ in range(30):
                        if not P.git_children(pid):break
                        time.sleep(1)
                    else:raise RuntimeError('CoordinatorGit still active')
                    P.git('add','-f','--',*files);P.git('commit','--only','-m',f"Archive 90M checkpoint {job['tag']} at {saved['seen']} presentations",'--',*files)
                    for _ in range(6):
                        P.git('pull','--rebase')
                        try:P.git('push','origin','main');break
                        except Exception:time.sleep(5)
                    else:raise RuntimeError('Checkpointpublication retriesexhausted')
                    state['published'].append(dict(tag=job['tag'],presented_targets=saved['seen'],files=files));save()
                except Exception as error:
                    state.update(status='needs_review',error=str(error));save();raise
                finally:os.kill(pid,signal.SIGCONT)
        time.sleep(30)


if __name__=='__main__':main()
