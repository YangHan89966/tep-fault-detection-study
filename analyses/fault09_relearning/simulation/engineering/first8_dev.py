"""Historical record of the 8 authorized first development simulations. Not a general-purpose runner."""
from pathlib import Path
import csv,json,sys,time
from concurrent.futures import ProcessPoolExecutor,as_completed
base=Path(__file__).resolve().parents[2];sys.path.insert(0,str(base/'simulation'))
from scientific_runner import run_row,sha,now,writej
fixed='be482fddc09941ed45b306da3f3d4a266812111c923cc564e0973b55ce3a7c99'
cases=['R01_pca_normal_001','R01_pca_normal_002','R01_normal_design_001','R01_normal_design_002','R01_fault_fit_001','R01_fault_fit_002','R01_fault_fit_003','R01_fault_select_001']
out=base/'simulation/engineering/FORMAL_START.json'
def main():
    l=base/'locks'; assert sha(l/'DESIGN_LOCK.json')==fixed;d=json.loads((l/'DESIGN_LOCK.json').read_text())
    assert all(sha(l/p)==h for p,h in d['files'].items())
    gate=base/'simulation/engineering/PORTABLE_ENGINE_QA.json';q=json.loads(gate.read_text()); assert q['pass'] and q['duration_hours']==48 and all(q['exact_array_bytes'].values()) and all(q['exact_full_run_digests'].values())
    with (l/'CASE_MANIFEST.csv').open(encoding='utf-8-sig') as f:allrows=list(csv.DictReader(f))
    assert len(allrows)==len({r['key_hex'] for r in allrows})==len({r['case_id'] for r in allrows})==2719
    rows=[r for r in allrows if r['case_id'] in cases];assert len(rows)==8 and all(r['stage']=='development' for r in rows)
    report=dict(stage='locked development only; no model or evaluation endpoints',started_utc=now(),design_lock_sha256=fixed,manifest_sha256=d['files']['CASE_MANIFEST.csv'],protocol_sha256=d['files']['PROTOCOL.json'],engineering_gate_sha256=sha(gate),workers=8,planned_case_ids=cases,completed=[])
    if out.is_file():raise RuntimeError('Historical kickoff is not repeatable: use main study resume driver.')
    t=time.perf_counter();writej(out,report)
    with ProcessPoolExecutor(8) as pool:
        fs=[pool.submit(run_row,r,base/'study/raw',fixed) for r in rows]
        for fut in as_completed(fs):
            rec=fut.result();p=base/'study/raw'/rec['run_id']/'COMPLETE.json';report['completed'].append(dict(case_id=rec['run_id'],key_hex=rec['key_hex'],complete_sha256=sha(p),raw_sha256=rec['outputs']['RAW.npz'],run_info_sha256=rec['outputs']['RUN_INFO.json'],wall_seconds=rec['wall_seconds'],rng_draw_count=rec['rng_draw_count'],shutdown_occurred=rec['shutdown_occurred']));report['n_completed']=len(report['completed']);writej(out,report)
    report['elapsed_seconds']=time.perf_counter()-t;report['finished_utc']=now();report['n_calibration']=report['n_evaluation']=0;writej(out,report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
