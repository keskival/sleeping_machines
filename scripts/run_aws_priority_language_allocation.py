"""Revision2 source-frozen AWS90M route-credit admission and full artifact publication."""
import argparse, datetime, fcntl, hashlib, json, math, os, subprocess, time, threading
from concurrent.futures import ThreadPoolExecutor
from run_aws_depth8_replay_matrix import validate_result
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args()
    os.chdir(ROOT);manifest=Path(a.manifest);plan=json.loads(manifest.read_text())
    state_path=manifest.parent/'worker_recovery.status.json'
    mutex=threading.RLock();stop_admission=threading.Event()
    if state_path.exists():raise RuntimeError('Existing lifecycle; preserve it and create a recovery plan')
    state=dict(status='waiting_host_lock',completed=[],current=None,active_streaming={},host=os.uname().nodename,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    def save():
        with mutex:
            temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(state_path)
    def frozen():
        for n,h in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+n)
    def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
    def publish(files,message):
        with open('/tmp/aws-language-publication.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            git('add','-f','--',*files);git('commit','--only','-m',message,'--',*files)
            for _ in range(6):
                git('pull','--rebase')
                if subprocess.run(['git','push','origin','main']).returncode==0:break
                time.sleep(5)
            else:raise RuntimeError('Publication retries exhausted')
    save();frozen()
    if os.uname().nodename!=plan['host']:raise RuntimeError('Assigned AWS host required')
    with open('/tmp/experiments-runner.lock','a') as reservation:
        # Wait for the current immutable three-slot matrix to release its reservation.
        fcntl.flock(reservation,fcntl.LOCK_EX)
        try:
            state['status']='running';save();pilots={}
            def stream_worker(slot, jobs):
                for job in jobs:
                    if stop_admission.is_set():return
                    frozen()
                    env=dict(os.environ,AWS_GYM_SLOT=str(slot),AWS_GYM_HOST_LOCK_FD=str(reservation.fileno()),
                             MEM_CAP_KB='6000000',MEM_CAP_RSS_KB='2000000',MIN_AVAIL_MB='8192',
                             JOB_TIMEOUT_S=str(job['timeout_s']))
                    with mutex:
                        state.setdefault('active_streaming',{})[str(slot)]=job['tag'];save()
                    try:
                        subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],env=env,
                            pass_fds=(reservation.fileno(),),check=True)
                        validate_result(ROOT,job)
                        files=[job['result']]
                        checkpoint=Path(job['result']).with_suffix('.progress.pt')
                        if checkpoint.exists():files.append(str(checkpoint))
                        publish(files,'Complete preserved streaming '+job['tag'])
                        with mutex:
                            state['completed'].append(job['tag']);state['active_streaming'].pop(str(slot),None);save()
                    except BaseException as error:
                        stop_admission.set()
                        with mutex:state.setdefault('background_errors',[]).append(str(error));save()
                        raise
            pool=ThreadPoolExecutor(max_workers=2)
            background=[pool.submit(stream_worker,int(slot),jobs) for slot,jobs in plan['streaming_slots'].items()]
            for job in plan['admission_jobs']:
                if stop_admission.is_set():raise RuntimeError('Background streaming failure; stop new admission')
                frozen()
                if job['stage']=='fit':
                    pilot=pilots[job['arm']];rate=pilot['train_chars_per_s']
                    if not math.isfinite(rate) or rate<=0:raise RuntimeError('Invalid pilot throughput')
                    timeout=math.ceil(1.5*94_000_000/rate+3600)
                    rss=max(2_000_000,math.ceil(pilot['max_rss_kb']*1.5))
                    if rss>6_000_000:raise RuntimeError('Pilot RSS requires reviewed admission above6GB')
                else:timeout,rss=1200,6_000_000
                env=os.environ.copy();env.update(AWS_GYM_SLOT='1',AWS_GYM_HOST_LOCK_FD=str(reservation.fileno()),
                    MEM_CAP_KB='24000000',MEM_CAP_RSS_KB=str(rss),MIN_AVAIL_MB='8192',JOB_TIMEOUT_S=str(timeout),
                    TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1')
                state['current']=dict(job=job,timeout_s=timeout,rss_cap_kb=rss);save()
                subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],env=env,
                    pass_fds=(reservation.fileno(),),check=True)
                queue=Path(job['queue'])
                files=[str(queue.parent/('runner_'+queue.stem+'.out')),
                       'experiments/queue/logs/'+job['name']+'.log']
                if job.get('result'):
                    result=json.loads(Path(job['result']).read_text())
                    assert result['status']=='completed' and result['args']['tag']==job['tag']
                    assert result['args']['route_credit']=='linear' and result['args']['compiled']
                    for n,h in result['source_sha256'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
                    files.append(job['result'])
                    weights=Path(result['final_weights'])
                    if not weights.is_file():raise RuntimeError('Completed run missing final weights')
                    files.append(str(weights))
                    if job['stage']=='pilot':
                        assert result['args']['max_windows']==60 and result['fitting_chars']==60*64*128
                        pilots[job['arm']]=result
                    else:
                        assert result['args']['fit']==90_000_000 and not result['args']['max_windows']
                        assert math.isfinite(result['dev_bpc']) and math.isfinite(result['test_bpc'])
                        checkpoint=Path('experiments/results/language_batched/checkpoints')/(job['tag']+'.pt')
                        if checkpoint.exists():files.append(str(checkpoint))
                state['completed'].append(job['name']);save()
                if files:publish(files,'Publish AWS 90M '+job['stage']+' '+job['name'])
            for future in background:future.result()
            pool.shutdown(wait=True)
            state.update(status='completed',current=None);save()
        except BaseException as error:
            stop_admission.set();state.update(status='needs_review',error=str(error));save()
            if 'pool' in locals():pool.shutdown(wait=True)
            raise


if __name__=='__main__':main()
