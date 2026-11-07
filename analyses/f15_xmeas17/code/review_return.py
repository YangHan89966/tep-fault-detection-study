"""Recalculate the paper table and descriptive timing diagnostics from the returned per-run records.

Requires Python 3.10+ (standard library only). Run in the extracted results package:
  python code/review_return.py
No original trajectories, simulation or model fitting are required for this results check.
"""
from pathlib import Path
import csv, hashlib, io, json, statistics as stat, zipfile
from collections import Counter

root=Path(__file__).resolve().parents[1]
out=root/'results'
out.mkdir(exist_ok=True)
archives=list((root/'source').glob('F15_XMEAS17_COMPLETE_*.zip'))
assert len(archives)==1
def table(z,name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8-sig'))))
def save(name,rows):
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    with zipfile.ZipFile(archives[0]) as z:
        assert z.testzip() is None
        manifest=table(z,'FILE_MANIFEST.csv')
        assert set(i['name'] for i in manifest)==set(z.namelist())-{'FILE_MANIFEST.csv'}
        for a in manifest:
            b=z.read(a['name']);assert int(a['bytes'])==len(b) and hashlib.sha256(b).hexdigest()==a['sha256']
        status=json.loads(z.read('STATUS.json'))
        assert status['status']=='COMPLETE' and status['matched']==240 and status['score_rows']==1920
        exact=table(z,'VERIFIED_INPUTS.csv')
        assert len({x['case_id'] for x in exact})==len({x['expected_sha256'] for x in exact})==240
        assert all(x['verified']=='True' for x in exact)
        per=table(z,'PER_RUN.csv')
        assert len({(x['case_id'],x['arm'],x['method']) for x in per})==len(per)==1920
        ss={(x['method'],x['arm']):x for x in table(z,'ARM_SUMMARY.csv')}
        exported=table(z,'INPUT_DIFFERENCES.csv')
        assert len(exported)==240 and all(int(x['rows'])==960 and float(x['max_abs_XMV_difference'])==0 for x in exported)
        for x in per:
            f=int(x['first']);post=int(x['first_post'])
            assert 0<=f<=960 and (post==0 or 161<=post<=960)
            assert x['role'] in ('normal','fault')
            assert int(x['prefault'])==int(0<f<=160)
            assert int(x['any_alarm'])==int(f>0)
            assert (int(x['clean40']),int(x['clean8']),int(x['clean2']))==(int(f>=161),int(161<=f<=321),int(161<=f<=201))
            assert int(x['loss_min'])==(3*(f-161) if f>=161 else 2400)
        pape=[];diagn=[]
        for method,m in [('MEWMC','Multivariate covariance chart (MEWMC, lambda=0.01)'),('full_pool','Supervised full-pool monitor (second-moment directions, squared)')]:
            for arm,base,source in [('ci','Controller-input','Controller-input'),('ci_with_post17','Controller-input','Post-update'),('post','Post-update','Post-update'),('post_with_ci17','Post-update','Controller-input')]:
                sample=[x for x in per if x['method']==method and x['arm']==arm]
                fx=[x for x in sample if x['role']=='fault'];nx=[x for x in sample if x['role']=='normal']
                assert len(fx)==len(nx)==120
                metrics=dict(clean40=sum(int(x['first'])>=161 for x in fx),clean8=sum(161<=int(x['first'])<=321 for x in fx),
                             clean2=sum(161<=int(x['first'])<=201 for x in fx),fault_prefault=sum(0<int(x['first'])<=160 for x in fx),
                             normal_any=sum(int(x['first'])>0 for x in nx),mean_loss_min=stat.mean(int(x['loss_min']) for x in fx))
                assert all(abs(float(ss[method,arm][k])-val)<1e-9 for k,val in metrics.items())
                pape.append(dict(monitor=m,base_recording=base,source_of_XMEAS17=source,n_fault=120,n_normal=120,**metrics))
                for role,grp in [('fault',fx),('normal',nx)]:
                    first=[int(x['first']) for x in grp if int(x['first'])>0]
                    counts=Counter(int(x['first_post']) for x in grp)
                    pk=[float(x['run_max'])/float(x['h']) for x in grp]
                    diagn.append(dict(method=method,arm=arm,role=role,n=len(grp),n_any_alarm=len(first),
                                      min_first_record_if_alarm=min(first) if first else '',
                                      median_first_record_if_alarm=stat.median(first) if first else '',
                                      max_first_record_if_alarm=max(first) if first else '',
                                      any_cross_after_record160=len(grp)-counts[0],first_post_cross_exactly_record161=counts[161],
                                      median_run_peak_over_limit=stat.median(pk)))
        save('Table_S64_proposed_F15_XMEAS17.csv',pape)
        save('FIRST_ALARM_DIAGNOSTICS.csv',diagn)
        for name in ('PAIRED_CONTRASTS.csv','ARM_SUMMARY.csv'):
            (out/name).write_bytes(z.read(name))
        print('All eight arm summaries matched the original per-run outcomes.')
if __name__=='__main__':main()
