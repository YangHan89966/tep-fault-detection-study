"""Extract signed-CUSUM chart extrema without re-running the process or refitting."""
from pathlib import Path
import argparse,csv,io,hashlib,json,zipfile
import numpy as np

def write(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(b):return hashlib.sha256(b).hexdigest()
def main(ref,normal,f03,f15,out):
    ref=Path(ref);out=Path(out);out.mkdir(exist_ok=True,parents=True)
    fi={r['path']:r for r in read(ref/'FILE_MANIFEST.csv')}
    cases={r['run_id']:r for r in read(ref/'MANIFEST.csv')}
    event={(r['run_id'],r['arm_id']):r for r in read(ref/'per_run_events.csv')}
    ms=json.loads((ref/'ARM_METADATA.json').read_text())
    lock=json.loads((ref/'THRESHOLDS_LOCK.json').read_text())
    calb=(ref/'CALIBRATION_MAXIMA.npz').read_bytes()
    assert sha(calb)==lock['calibration_maxima_sha256']
    cal=np.load(io.BytesIO(calb),allow_pickle=False)['maxima']
    assert np.array_equal(np.sort(cal,axis=0)[189],np.asarray(lock['thresholds']))
    selected=[m for m in ms if m['study']=='R5_fixed' and m['fault'] in (3,15)]
    assert len(selected)==20
    def label(m):
        c1=m['fault']==3
        fav=(m['choice_key']=='mean_pool_signed') if c1 else (m['choice_key']=='moment_pool_squared')
        return dict(panel='confirmation',contrast='C1' if c1 else 'C3',fault=m['fault'],replicate=m['rep'],arm='favoured' if fav else 'comparator',source_method=m['arm_id'])
    rows=[];crows=[];checked=0;src=[]
    for m in selected:
        col=m['column']
        for i,cid in enumerate(lock['calibration_run_ids']):
            crows.append(dict(**label(m),run_id=i+1,case_id=cid,original_h=lock['thresholds'][col],cal48_max=cal[i,col]))
    for ar,ff,n in [(normal,0,400),(f03,3,200),(f15,15,200)]:
        p=Path(ar); archive_hash=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
        with zipfile.ZipFile(p) as z:
            members=sorted(nm for nm in z.namelist() if nm.endswith('/CHARTS.npz'));assert len(members)==n
            for nm in members:
                cid=nm.split('/')[-2];case=cases[cid];assert int(case['fault_id'])==ff
                b=z.read(nm); assert sha(b)==fi['runs/'+cid+'/CHARTS.npz']['sha256']
                with np.load(io.BytesIO(b),allow_pickle=False) as a:charts=a['charts']
                assert charts.shape==(960,22) and np.isfinite(charts).all()
                for m in selected:
                    if ff and ff!=m['fault']:continue
                    col=m['column'];h=lock['thresholds'][col];s=charts[:,col]
                    pre,p8,p40,all48=s[:160].max(),s[160:321].max(),s[160:].max(),s.max()
                    old=event[cid,m['arm_id']]
                    for metric,bv in [('prefault_alarm',pre>h),('clean40',pre<=h and p40>h),('clean8',pre<=h and p8>h)]:
                        assert int(old[metric])==int(bv),(cid,m['arm_id'],metric,old[metric],bv)
                    if ff==0:assert int(old['normal_alarm'])==int(all48>h)
                    checked+=1
                    rows.append(dict(**label(m),role='normal' if ff==0 else 'fault',run_id=int(case['index']),case_id=cid,original_h=h,
                                     pre_max=pre,post8_max=p8,post40_max=p40,normal48_max=all48 if ff==0 else ''))
        src.append(dict(file=p.name,sha256=archive_hash,charts_checked=n))
    assert len(rows)==12000 and len(crows)==3980
    write(out/'run_peaks.csv',rows);write(out/'calibration_peaks.csv',crows)
    (out/'AUDIT.json').write_text(json.dumps(dict(n_chart_arrays=800,run_arm_rows_checked=checked,calibration_estimates_per_arm=199,source_archives=src,
        chart_shape=[960,22],calibration_maxima_sha256=sha(calb),original_alarms_replayed=True,
        rule='Strict > h; all trajectories reset at beginning only; early post window records 161 through321 inclusive.'),indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--normal',required=True);p.add_argument('--f03',required=True);p.add_argument('--f15',required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.reference,a.normal,a.f03,a.f15,a.out)
