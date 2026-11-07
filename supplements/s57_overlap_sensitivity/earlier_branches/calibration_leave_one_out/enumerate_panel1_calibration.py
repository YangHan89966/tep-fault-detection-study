#!/usr/bin/env python3
"""Enumerate all 59 single-calibration-run deletions on the same A4 test subset.
Only saved scores/maxima are used; no fit, simulation, bootstrap or Panel 2 work.
"""
import argparse, hashlib, io, json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd

MONITORS=[('M1','controller-input','second-moment'),('M2','native','second-moment'),
          ('M3','controller-input','mean-shift'),('M4','native','mean-shift')]
PANEL='S25_primary'

def sha(data):return hashlib.sha256(data).hexdigest()
def chart(score):
    assert score.ndim==3 and score.shape[1:]==(960,4) and np.isfinite(score).all()
    p=np.zeros((len(score),4));m=p.copy();out=np.empty_like(score)
    for t in range(960):
        p=np.maximum(0.,p+score[:,t,:]-.05);m=np.maximum(0.,m-score[:,t,:]-.05)
        out[:,t,:]=np.maximum(p,m)
    return out
def limit(peaks,anchors,rank):return anchors*np.sort(peaks/anchors,axis=0)[rank-1]
def endpoints(stat,h,fault,normal):
    mask=stat>h;pre=mask[:,:160].any(1);post=mask[:,160:]
    detected=post.any(1);delay=np.where(detected,post.argmax(1)*3,2400)
    clean=detected&~pre;loss=np.where(clean,delay,2400)
    return [dict(threshold=float(h[j]),fault_n=int(fault.sum()),normal_n=int(normal.sum()),
                 clean40_n=int(clean[fault,j].sum()),clean8_n=int((clean[fault,j]&(delay[fault,j]<=480)).sum()),
                 mean_loss_min=float(loss[fault,j].mean()),normal_alarm_n=int(mask[normal,:,j].any(1).sum()),
                 FAR_pct=float(mask[normal,:,j].any(1).mean()*100)) for j in range(4)]

def main(archive,inventory,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);sources=[]
    def record(data,name):
        sources.append(dict(source=name,bytes=len(data),sha256=sha(data)));return data
    archive=Path(archive).resolve();inventory=Path(inventory).resolve()
    inv=pd.read_csv(io.BytesIO(record(inventory.read_bytes(),str(inventory))))
    inv=inv[inv.panel=='recording_panel1'].set_index('run_id')
    assert inv.index.is_unique and len(inv)==194 and (inv.duration_hours==48).all()
    assert np.array_equal(inv.self_wrap,inv.seed_u32%32==0)
    archive_bytes=record(archive.read_bytes(),str(archive))
    with ZipFile(io.BytesIO(archive_bytes)) as outer:
        member_name=next(n for n in outer.namelist() if n.endswith('F9_RELEARN_RANK_20261007T035904Z_ad63023a_RETURN.zip'))
        inner=record(outer.read(member_name),str(archive)+'!'+member_name)
    with ZipFile(io.BytesIO(inner)) as source:
        def member(name):return record(source.read(name),str(archive)+'!'+member_name+'!'+name)
        proto=json.loads(member('PROTOCOL.json'))
        assert [(x['id'],x['view'],x['pool']) for x in proto['monitors']]==[
            ('M1','snapshot','moment'),('M2','native','moment'),('M3','snapshot','mean'),('M4','native','mean')]
        lock_bytes=member('workspace/THRESHOLDS_LOCK.json');lock=json.loads(lock_bytes)
        final=json.loads(member('workspace/FINAL_LOCK.json'))
        assert final['threshold_lock_sha256']==sha(lock_bytes)
        calibration_bytes=member('workspace/'+lock['relative_path']+'/'+PANEL+'.npz')
        assert sha(calibration_bytes)==lock['files'][PANEL+'.npz']
        with np.load(io.BytesIO(calibration_bytes),allow_pickle=False) as a:cal={k:a[k] for k in a.files}
        evaluation_bytes=member('workspace/'+final['relative_path']+'/'+PANEL+'_EVALUATION.npz')
        assert sha(evaluation_bytes)==final['files'][PANEL+'_EVALUATION.npz']
        with np.load(io.BytesIO(evaluation_bytes),allow_pickle=False) as a:ev={k:a[k] for k in a.files}
        cases_bytes=member('workspace/'+final['relative_path']+'/'+PANEL+'_CASES.json')
        assert sha(cases_bytes)==final['files'][PANEL+'_CASES.json']
        cases=json.loads(cases_bytes)
    cal_ids=lock['panels'][PANEL]['case_ids'];ci=inv.loc[cal_ids]
    ei=inv.loc[[c['case_id'] for c in cases]]
    assert len(ci)==59 and (ci.role=='normal_calibration').all() and len(ei)==135
    assert ei.seed_u32.tolist()==[c['seed_u32'] for c in cases]
    assert ei.role.tolist()==[c['role'] for c in cases]
    assert not set(ci.index)&set(ei.index)
    assert ci.self_wrap.sum()==1
    target_index=int(np.flatnonzero(ci.self_wrap.to_numpy())[0]);target_case=cal_ids[target_index]
    keep=~ei.self_wrap.to_numpy()
    fk=keep&(ei.role.to_numpy()=='fault9_evaluation');nk=keep&(ei.role.to_numpy()=='normal_evaluation')
    assert fk.sum()==70 and nk.sum()==58
    assert cal['peaks'].shape==(59,4) and cal['anchors'].shape==(4,) and (cal['anchors']>0).all()
    original_h=limit(cal['peaks'],cal['anchors'],57)
    assert np.array_equal(original_h,cal['thresholds']) and np.array_equal(original_h,ev['thresholds'])
    stat=chart(ev['scores'])
    # One fixed-point check is needed to use saved scores for the new thresholds.
    assert np.array_equal(stat>original_h,np.unpackbits(ev['alarm_packed'],axis=1,count=960,bitorder='little').astype(bool))
    baseline=endpoints(stat,original_h,fk,nk)
    assert [x['clean40_n'] for x in baseline]==[70,25,66,10]
    assert [x['clean8_n'] for x in baseline]==[48,1,21,1]
    assert [x['normal_alarm_n'] for x in baseline]==[3,2,1,1]
    for row,(mid,view,pool) in zip(baseline,MONITORS):row.update(monitor=mid,recording=view,pool=pool,calibration_n=59,rank=57)
    rows=[]
    new_rank=(95*(58+1)+99)//100;assert new_rank==57
    for deleted in range(59):
        cm=np.arange(59)!=deleted;h=limit(cal['peaks'][cm],cal['anchors'],new_rank)
        result=endpoints(stat,h,fk,nk)
        for j,((mid,view,pool),r) in enumerate(zip(MONITORS,result)):
            p=cal['peaks'][deleted,j]
            row=dict(deleted_case_id=cal_ids[deleted],deleted_seed_u32=int(ci.seed_u32.iloc[deleted]),
                     deleted_case_self_overlapping=bool(ci.self_wrap.iloc[deleted]),
                     monitor=mid,recording=view,pool=pool,calibration_n=58,rank=57,
                     deleted_peak_original_rank_min=int((cal['peaks'][:,j]<p).sum()+1),
                     deleted_peak_original_rank_max=int((cal['peaks'][:,j]<=p).sum()),
                     deleted_peak=float(p),original_threshold=float(original_h[j]),
                     threshold_change=float(h[j]-original_h[j]),**r)
            rows.append(row)
    table=pd.DataFrame(rows);assert len(table)==236
    metrics=['threshold','clean40_n','clean8_n','mean_loss_min','normal_alarm_n','FAR_pct']
    metric_rows=[];outcome_rows=[];target_rows=[]
    for j,(mid,view,pool) in enumerate(MONITORS):
        x=table[table.monitor==mid];target=x[x.deleted_case_self_overlapping].iloc[0]
        assert len(x)==59
        for metric in metrics:
            for value,number in x.groupby(metric,sort=True).size().items():
                metric_rows.append(dict(monitor=mid,metric=metric,value=float(value),deletion_count=int(number),
                                        fraction_of_59=float(number/59),is_self_overlap_deletion_value=bool(value==target[metric])))
        for values,number in x.groupby(metrics,sort=True).size().items():
            outcome_rows.append(dict(monitor=mid,**dict(zip(metrics,values)),deletion_count=int(number),
                                     fraction_of_59=float(number/59)))
        r=dict(monitor=mid,recording=view,pool=pool,target_deleted_case=target_case,
               target_deleted_seed_u32=int(target.deleted_seed_u32),
               target_peak_original_rank_min=int(target.deleted_peak_original_rank_min),
               target_peak_original_rank_max=int(target.deleted_peak_original_rank_max),
               original_threshold=float(original_h[j]),target_threshold=float(target.threshold),
               target_threshold_increase=float(target.threshold_change),
               deletions_with_any_threshold_increase=int((x.threshold_change>0).sum()),
               deletions_same_threshold_as_target=int((x.threshold==target.threshold).sum()),
               deletions_same_clean40_as_target=int((x.clean40_n==target.clean40_n).sum()),
               deletions_same_clean8_as_target=int((x.clean8_n==target.clean8_n).sum()),
               deletions_same_FAR_as_target=int((x.normal_alarm_n==target.normal_alarm_n).sum()),
               deletions_same_clean40_and8_as_target=int(((x.clean40_n==target.clean40_n)&(x.clean8_n==target.clean8_n)).sum()),
               deletions_same_all_requested_outcomes_as_target=int((x[metrics]==target[metrics]).all(axis=1).sum()),
               baseline_clean40_n=baseline[j]['clean40_n'],target_clean40_n=int(target.clean40_n),
               baseline_clean8_n=baseline[j]['clean8_n'],target_clean8_n=int(target.clean8_n),
               baseline_loss_min=baseline[j]['mean_loss_min'],target_loss_min=float(target.mean_loss_min),
               normal_alarm_n=int(target.normal_alarm_n),fault_n=70,normal_n=58)
        target_rows.append(r)
    # Each row is one shared physical deletion, never four independently chosen deletions.
    joint=table.pivot(index='deleted_case_id',columns='monitor',values='threshold')
    patterns=joint.groupby(['M1','M2','M3','M4'],sort=True).size().rename('deletion_count').reset_index()
    table.to_csv(out/'LEAVE_ONE_OUT_BY_CASE_MONITOR.csv',index=False)
    pd.DataFrame(metric_rows).to_csv(out/'DISTINCT_METRIC_FREQUENCIES.csv',index=False)
    pd.DataFrame(outcome_rows).to_csv(out/'JOINT_OUTCOME_FREQUENCIES_BY_MONITOR.csv',index=False)
    pd.DataFrame(target_rows).to_csv(out/'SELF_OVERLAP_DELETION_COMPARISON.csv',index=False)
    pd.DataFrame(baseline).to_csv(out/'ORIGINAL_CALIBRATION_ON_FIXED_A4_TEST_SUBSET.csv',index=False)
    patterns.to_csv(out/'SHARED_DELETION_THRESHOLD_PATTERNS.csv',index=False)
    case_table=inv.loc[cal_ids+[c['case_id'] for c in cases]].reset_index()
    case_table['excluded_from_fixed_evaluation_subset']=case_table.role.ne('normal_calibration')&case_table.self_wrap
    case_table.to_csv(out/'PANEL1_RUN_IDENTITIES.csv',index=False)
    (out/'INPUT_HASHES.json').write_text(json.dumps(sources,indent=2)+'\n')
    status=dict(status='COMPLETE_EXACT_ENUMERATION',date='2026-10-09',panel='Panel1 only',
                enumerated_physical_deletions=59,monitors_per_deletion=4,rows=236,
                fixed_fault_evaluation_n=70,fixed_normal_evaluation_n=58,
                original_calibration_n=59,remaining_calibration_n=58,rank_before=57,rank_after=57,
                rank_rule='ceil(.95*(n+1))',strict_greater_alarm=True,original_anchor_and_models_fixed=True,
                original_threshold_reconstruction_equal=True,original_alarm_mask_reconstruction_equal=True,
                baseline_A4_test_subset_counts_equal=True,self_overlapping_calibration_case=target_case,
                bootstrap_draws=0,new_simulations=0,refits=0,original_file_edits=0,
                frequencies_are_exact_counts_of_this_59_case_enumeration_not_sampling_probabilities=True,
                independence_established=False)
    (out/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    rank_arithmetic=[dict(scenario=name,calibration_n=n,rank_1based=57,
                          nominal_tail_target=.05,tail_numerator=n+1-57,tail_denominator=n+1,
                          formal_tail_probability=(n+1-57)/(n+1),
                          formal_tail_percent=100*(n+1-57)/(n+1),
                          assumptions='continuous and exchangeable calibration/test run scores',
                          empirical_FAR_guarantee_for_this_dataset=False)
                     for name,n in [('original_59',59),('leave_one_out_58',58)]]
    pd.DataFrame(rank_arithmetic).to_csv(out/'FINITE_SAMPLE_RANK_ARITHMETIC.csv',index=False)
    lines=['Panel1校准留一条运行：59种精确枚举，2026-10-09',
           '固定原模型、anchor与A4双排除测试子集（70故障、58正常），每次四监视器删除同一物理校准运行。',
           '59→58条校准时ceil[.95(n+1)]的秩仍是57；无模拟、拟合或bootstrap。',
           '原59条阈值、原报警位及固定测试子集计数均恢复。',
           f'被标记的自重叠校准运行：{target_case}，seed={int(ci.seed_u32.iloc[target_index])}。','']
    for r in target_rows:
        lines.append(f"{r['monitor']}：该运行原峰值升序秩{r['target_peak_original_rank_min']}；h {r['original_threshold']:.9f}→{r['target_threshold']:.9f}。相同新h出现{r['deletions_same_threshold_as_target']}/59次；相同40h/8h联合检出结果出现{r['deletions_same_clean40_and8_as_target']}/59次。")
        lines.append(f"  clean40 {r['baseline_clean40_n']}/70→{r['target_clean40_n']}/70；clean8 {r['baseline_clean8_n']}/70→{r['target_clean8_n']}/70；loss {r['baseline_loss_min']:.6f}→{r['target_loss_min']:.6f}min；新h正常报警{r['normal_alarm_n']}/58。")
    lines+=['','所有distinct阈值、clean40、clean8、loss及FAR的次数见DISTINCT_METRIC_FREQUENCIES.csv；其联合组合见JOINT_OUTCOME_FREQUENCIES_BY_MONITOR.csv。',
            '这些比例是对本59条校准运行的精确枚举频数，不是校准总体中的概率或置信区间。',
            '解释：当原峰值没有并列，删去升序第1–57名中的任一条后，第57个保留峰值是原第58个峰值；删原第58或59名，则仍取原第57个峰值。这一有限校准尾部的秩变化可产生普遍的阈值上跳，而非自重叠标签独有的异常。',
            '秩的离散性：原n=59、r=57时，形式尾部为(60−57)/60=5%；n=58、r=57时为(59−57)/59=2/59≈3.3898%。因此沿用ceil[.95(n+1)]规则会跳到更保守的可达秩。本轮未改变阈值规则。',
            '上述尾部表达式是连续、可交换的校准与测试运行得分前提下的有限样本秩算术，不是本数据的FAR保证；当前随机流重叠问题并未因此消除。算术单列于FINITE_SAMPLE_RANK_ARITHMETIC.csv，与实际正常FAR及59种删除的枚举频数分别报告。',
            '该解释不消除实际校准敏感性：M2在同一测试子集上的检出下降依然真实存在。它也不证明不同运行独立，不能替代对随机流重叠或校准总体不确定性的分析。',
            '复算：python enumerate_panel1_calibration.py --relearn-archive <TEP_F9_Relearn_Rank_Final_Analysis_20261007.zip> --inventory <RUN_SEED_CYCLES_AND_ACTUAL_DRAWS.csv> --out <新目录>']
    (out/'SUMMARY_CN.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pd.DataFrame(target_rows)[['monitor','target_deleted_case','target_peak_original_rank_min',
          'original_threshold','target_threshold','deletions_same_threshold_as_target',
          'baseline_clean40_n','target_clean40_n','baseline_clean8_n','target_clean8_n',
          'deletions_same_clean40_as_target','deletions_same_clean40_and8_as_target']].to_string(index=False))
    print(pd.DataFrame(outcome_rows).to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--relearn-archive',type=Path,required=True)
    p.add_argument('--inventory',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.relearn_archive,a.inventory,a.out)
