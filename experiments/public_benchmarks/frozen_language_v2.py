"""Resume source-frozen language runs without undoing another host's core edits."""
import argparse,hashlib,importlib.util,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--compiled-source',required=True);p.add_argument('--compiled-sha',required=True)
    p.add_argument('--driver-source',required=True);p.add_argument('--driver-sha',required=True)
    p.add_argument('driver');args,rest=p.parse_known_args();source=ROOT/args.compiled_source
    assert hashlib.sha256(source.read_bytes()).hexdigest()==args.compiled_sha
    name='sleeping_machines.compiled_episodes';spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    driver_source=ROOT/args.driver_source;assert hashlib.sha256(driver_source.read_bytes()).hexdigest()==args.driver_sha
    sys.argv=[str(ROOT/args.driver),*rest];exec(compile(driver_source.read_bytes(),str(ROOT/args.driver),'exec'),dict(__file__=str(ROOT/args.driver),__name__='__main__'))
    tag=rest[rest.index('--tag')+1];out=ROOT/'experiments/results/language_batched'/(tag+'.json');record=json.loads(out.read_text());assert record['status']=='completed'
    disk=record['source_sha256'].pop(name.replace('.','/')+'.py')
    record['source_sha256'][args.compiled_source]=args.compiled_sha
    driver_disk=record['source_sha256'].pop(args.driver)
    record['source_sha256'][args.driver_source]=args.driver_sha
    record['source_sha256'][str(Path(__file__).relative_to(ROOT))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    record['execution_source_aliases']={'sleeping_machines/compiled_episodes.py':dict(actual_source=args.compiled_source,actual_sha256=args.compiled_sha,disk_sha256_at_driver_start=disk,
        reason='Source-exact recovery/continuation after another host added an optional feedback hook. Archived original is the module actually loaded. New shared source retained untouched.')}
    record['execution_source_aliases'][args.driver]=dict(actual_source=args.driver_source,actual_sha256=args.driver_sha,disk_sha256_at_driver_start=driver_disk)
    temp=out.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temp.replace(out)
