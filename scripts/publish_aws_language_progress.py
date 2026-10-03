"""Archive source-exact live language checkpoints at milestones, serializing Git publication."""
import argparse,datetime,fcntl,hashlib,json,os,signal,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import torch

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def coordinator(manifest):
    candidates=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:args=(path/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(name in args for name in (b'scripts/run_aws_depth8_replay_matrix.py',b'scripts/run_aws_priority_language_allocation.py',b'scripts/run_aws_public_campaign.py',b'scripts/run_aws_public_campaign_v2.py')) and str(manifest).encode() in args:candidates.append(int(path.name))
    if len(candidates)!=1:raise RuntimeError('One exact matching coordinator required')
    return candidates[0]
def git_children(pid):
    for task in (Path('/proc')/str(pid)/'task').iterdir():
        try:children=(task/'children').read_text().split()
        except OSError:continue
        for child in children:
            try:exe=Path('/proc')/child/'exe';name=exe.resolve().name
            except OSError:continue
            if name=='git':return True
    return False

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--interval-targets',type=int,default=250000);a=p.parse_args()
    os.chdir(ROOT)
    if git('branch','--show-current')!='main':raise RuntimeError('Main required')
    torch.set_num_threads(1);plan=json.loads(Path(a.manifest).read_text());directory=ROOT/'experiments/results/aws_language_progress';directory.mkdir(exist_ok=True)
    status=Path(a.manifest).parent/'progress_publisher.status.json';state=dict(status='monitoring',published=[])
    def save():status.write_text(json.dumps(state,indent=2)+'\n')
    seen={};save()
    while True:
        lifecycle=json.loads((Path(a.manifest).parent/'worker_recovery.status.json').read_text())
        if lifecycle['status'] in ('completed','needs_review'):
            state['status']='stopped_'+lifecycle['status'];save();return
        for job in plan['jobs']:
            if job['stage']!='pilot':continue
            checkpoint=Path(job['result']).with_suffix('.progress.pt')
            if not checkpoint.exists() or Path(job['result']).exists():continue
            mtime=checkpoint.stat().st_mtime_ns
            if seen.get(job['tag'])==mtime:continue
            seen[job['tag']]=mtime
            # Atomic rename in the trainer means this opens one consistent inode.
            saved=torch.load(checkpoint,weights_only=False);targets=int(saved['total_targets']);milestone=targets//a.interval_targets
            if milestone<1:continue
            archive=directory/f"{job['tag']}_milestone{milestone:03d}.pt";metadata=archive.with_suffix('.json')
            if archive.exists():continue
            temporary=archive.with_suffix('.tmp');torch.save(saved,temporary);temporary.replace(archive)
            record=dict(status='checkpoint',args=saved['result']['args'],trained_targets=targets,optimizer_updates=saved['updates'],cursor=saved['cursor'],learner_counters=saved.get('learner_counters'),protocol=saved['result']['protocol'],hardware=saved['result'].get('hardware'),cumulative_wall_s=saved['result'].get('wall_s'),source_sha256=saved['result']['source_sha256'],checkpoint=str(archive.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Partial learning checkpoint and exact cursor/RNG/private-state/Adam; online scores are not heldout or completed10M quality. Numerical arrays retained; no projected completed metric.')
            metadata.write_text(json.dumps(record,indent=2)+'\n');files=[str(archive.relative_to(ROOT)),str(metadata.relative_to(ROOT))]
            lock=open('/tmp/aws-language-publication.lock','a');fcntl.flock(lock,fcntl.LOCK_EX)
            pid=coordinator(a.manifest);os.kill(pid,signal.SIGSTOP)
            try:
                for _ in range(30):
                    if not git_children(pid):break
                    time.sleep(1)
                else:raise RuntimeError('Coordinator Git child still active')
                if (ROOT/'.git/rebase-merge').exists() or (ROOT/'.git/rebase-apply').exists():raise RuntimeError('Resolve active rebase before publication')
                git('add','-f','--',*files);git('commit','--only','-m',f"Archive language checkpoint {job['tag']} at {targets} targets",'--',*files)
                for attempt in range(6):
                    git('pull','--rebase')
                    result=subprocess.run(['git','push','origin','main'],cwd=ROOT)
                    if result.returncode==0:break
                    time.sleep(5)
                else:raise RuntimeError('Checkpoint push retries exhausted')
                state['published'].append(dict(tag=job['tag'],targets=targets,files=files));save()
            except Exception as error:
                state.update(status='needs_review',error=str(error));save();raise
            finally:
                os.kill(pid,signal.SIGCONT);lock.close()
        time.sleep(30)
if __name__=='__main__':main()
