"""Source-frozen three-slot continuation with a public benchmark priority lane."""
import argparse,datetime,fcntl,hashlib,json,math,os,subprocess,threading,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);a=p.parse_args();os.chdir(ROOT)
    path=Path(a.manifest);plan=json.loads(path.read_text());status=path.parent/'worker_recovery.status.json'
    if status.exists():raise RuntimeError('Existing lifecycle; new recovery manifest required')
    assert os.uname().nodename==plan['host']
    state=dict(status='waiting_host_lock',active={},completed=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    mutex=threading.RLock();failed=threading.Event();pilots={}
    def save():
        with mutex:
            temporary=status.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(status)
    def frozen():
        for name,digest in {**plan['source_sha256'],**plan['queue_sha256']}.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
    def publish(files,message):
        with open('/tmp/aws-language-publication.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            git('add','-f','--',*files);git('commit','--only','-m',message,'--',*files)
            for _ in range(6):
                git('pull','--rebase')
                if subprocess.run(['git','push','origin','main']).returncode==0:return
                time.sleep(5)
            raise RuntimeError('Publication retries exhausted')
    save();frozen()
    with open('/tmp/experiments-runner.lock','a') as reservation:
        fcntl.flock(reservation,fcntl.LOCK_EX);state['status']='running';save()
        def worker(slot,jobs):
            try:
                jobs=list(jobs);index=0
                while index<len(jobs):
                    job=jobs[index];index+=1
                    if failed.is_set():return
                    frozen();kind=job['kind'];timeout=job['timeout_s'];rss=job['rss_kb']
                    if kind=='language_fit' and job.get('pilot_tag'):
                        pilot=pilots[job['pilot_tag']];timeout=math.ceil(1.5*94_000_000/pilot['train_chars_per_s']+3600)
                        rss=max(2_000_000,math.ceil(pilot['max_rss_kb']*1.5));assert rss<=6_000_000
                    env=dict(os.environ,AWS_GYM_SLOT=str(slot),AWS_GYM_HOST_LOCK_FD=str(reservation.fileno()),
                             MEM_CAP_KB='24000000' if kind!='streaming' else '6000000',MEM_CAP_RSS_KB=str(rss),
                             MIN_AVAIL_MB='8192',JOB_TIMEOUT_S=str(timeout),TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1')
                    with mutex:state['active'][str(slot)]=dict(job=job,timeout_s=timeout,rss_kb=rss);save()
                    subprocess.run(['bash','experiments/queue/run_safe.sh',job['queue']],env=env,pass_fds=(reservation.fileno(),),check=True)
                    result=json.loads(Path(job['result']).read_text());assert result['status']=='completed'
                    for name,digest in result['source_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
                    files=[job['result'],'experiments/queue/logs/'+job['name']+'.log',str(Path(job['queue']).parent/('runner_'+Path(job['queue']).stem+'.out'))]
                    if kind.startswith('language'):
                        assert result['args']['tag']==job['tag'] and result['args']['compiled'] and result['args']['route_credit']=='linear'
                        if kind=='language_pilot':pilots[job['tag']]=result
                        else:assert result['fitting_chars']==89997312 and math.isfinite(result['dev_bpc'])
                        files.append(result['final_weights'])
                    elif kind=='streaming':
                        assert result['args']['tag']==job['tag'];files.append(str(Path(job['result']).with_suffix('.progress.pt')))
                    elif kind in ('public','public_final'):
                        assert result['args']['tag']==job['tag'] 
                        if kind=='public':assert result['args']['stage']=='screen' and result['test'] is None
                        else:assert result['args']['stage']=='final' and result['test']['targets']=={'ECG200':100,'JapaneseVowels':370,'PenDigits':3498}[result['args']['dataset']]
                        files.append(result['checkpoint'])
                    if kind=='selection':
                        selected=json.loads(Path(result['selected_screen']).read_text());opts=selected['args'];following=[]
                        assert hashlib.sha256(Path(result['selected_screen']).read_bytes()).hexdigest()==result['selected_screen_sha256']
                        for seed in result['final_seeds']:
                            tag=job['tag']+'_final_s'+str(seed)
                            queue=path.parent/(tag+'.txt')
                            command='experiments/public_benchmarks/run.py --stage final --tag '+tag+' --dataset '+opts['dataset']+' --selection '+result['selected_screen']+' --seed '+str(seed)+' --epochs '+str(result['final_epochs'])
                            for key in ('payload','depth','pool','heads','lr','batch'):command+=' --'+key+' '+str(opts[key])
                            assert not queue.exists();queue.write_text(tag+' '+command+'\n')
                            following.append(dict(name=tag,tag=tag,kind='public_final',queue=str(queue),result='experiments/results/public_benchmarks/'+tag+'.json',timeout_s=7200,rss_kb=6_000_000))
                            files.append(str(queue))
                        jobs[index:index]=following
                    publish(files,'Publish guarded '+kind+' '+job['tag'])
                    with mutex:state['completed'].append(job['name']);state['active'].pop(str(slot),None);save()
            except BaseException as error:
                failed.set()
                with mutex:state.setdefault('errors',[]).append(dict(slot=slot,error=str(error)));save()
                raise
        with ThreadPoolExecutor(max_workers=3) as pool:
            tasks=[pool.submit(worker,int(slot),jobs) for slot,jobs in plan['slots'].items()]
            try:
                for task in tasks:task.result()
                state['status']='completed';save()
            except BaseException:
                state['status']='needs_review';save();raise

if __name__=='__main__':main()
