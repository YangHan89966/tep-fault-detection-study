"""D3: denominator controls from the full saved candidate results; no re-fitting."""
from common import *
import zipfile,collections,argparse

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path);args=ap.parse_args()
 src=args.source or Path(CONTRACT['sources']['s28_results']['path']);assert sha(src)==CONTRACT['sources']['s28_results']['sha256']
 out=ROOT/'results/D3'; rows=[]; selections=[]; field_checks=0; counts=[]
 with zipfile.ZipFile(src) as z:
  for variant in CONTRACT['D3']['variants']:
   for si in range(1,11):
    split=f'D{si:02}';p=f'models/F09/{split}/{variant}/'
    ch=json.loads(z.read(p+'CHOICES.json'));n=ch['mean_count']+ch['moment_count'];km=ch['mean_count']
    inn=csv_rows(z.read(p+'INNER_SCORES.csv'));ev=csv_rows(z.read(p+'ALL_CANDIDATE_RESULTS.csv'))
    rec=read_npz(z,p+'EVALUATION_RECORDS.npz');fields=list(rec['fault_fields']);nfields=list(rec['normal_fields'])
    # Keep the exact table inputs needed to rerun this extraction.
    for fn in ['CHOICES.json','EIGEN_DIAGNOSTICS.csv','INNER_SCORES.csv','ALL_CANDIDATE_RESULTS.csv']:
     q=ROOT/'inputs/D3'/split/variant/fn;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(z.read(p+fn))
    mapping={}
    for j,labels in enumerate(ch['new_labels']):
     for label in labels:mapping[label]=km+j
    winning=ch['choices']['moment_pool_signed'];wa=winning%n
    counts.append(dict(variant=variant,split=split,candidate=ch['candidate_names'][winning],labels=';'.join(ch['new_labels'][wa-km])))
    for fami,fam in enumerate(FAMILIES):
     for metric in ['unweighted','shrink50','shrink10','raw']:
      options=[]
      for extreme in ['max','min']:
       ci=fami*n+mapping[f'{metric}_{extreme}'];r=ev[ci];assert int(r['candidate_index'])==ci
       rr=rec['fault'][ci];nn=rec['normal'][ci]
       success=rr[:,fields.index('clean_detection')];loss=rr[:,fields.index('event_loss_samples')]
       pre=rr[:,fields.index('prefault_alarm')];alarms=nn[:,nfields.index('alarm')]
       assert int(success.sum())==int(r['clean_detected_runs']);assert int(pre.sum())==int(r['prefault_alarm_runs'])
       assert abs(float(r['event_loss_min'])-3*float(loss.mean()))<1e-9
       assert int(alarms.sum())==int(r['normal_alarm_runs']);field_checks+=4
       row=dict(variant=variant,split=split,metric=metric,extreme=extreme,family=fam,candidate_index=ci,
        candidate=ch['candidate_names'][ci],inner_loss_min=float(inn[ci]['event_loss_min']),
        success_percent=100*float(success.mean()),event_loss_min=3*float(loss.mean()),
        normal_far_percent=100*float(alarms.mean()),prefault_runs=int(pre.sum()),fault_n=len(rr),normal_n=len(nn))
       rows.append(row);options.append(row)
      chosen=min(options,key=lambda r:(r['inner_loss_min'],r['candidate_index']))
      selections.append({**chosen,'selection_rule':'minimum original development loss among two extrema; original candidate-index tie order'})
 summary=[]
 for variant in CONTRACT['D3']['variants']:
  for metric in ['unweighted','shrink50','shrink10','raw']:
   for fam in FAMILIES:
    ss=[r for r in selections if r['variant']==variant and r['metric']==metric and r['family']==fam]
    row=dict(variant=variant,metric=metric,family=fam,max_chosen=sum(r['extreme']=='max' for r in ss),min_chosen=sum(r['extreme']=='min' for r in ss))
    for key in ['success_percent','event_loss_min','normal_far_percent','prefault_runs']:row.update(summary_stat([r[key] for r in ss],key))
    summary.append(row)
 extrema=[]
 for variant in CONTRACT['D3']['variants']:
  for metric in ['unweighted','shrink50','shrink10','raw']:
   for extreme in ['max','min']:
    for fam in FAMILIES:
     ss=[r for r in rows if r['variant']==variant and r['metric']==metric and r['extreme']==extreme and r['family']==fam]
     row=dict(variant=variant,metric=metric,extreme=extreme,family=fam)
     for key in ['success_percent','event_loss_min','normal_far_percent']:row.update(summary_stat([r[key] for r in ss],key))
     extrema.append(row)
 write_csv(out/'ALL_EXTREMA_PER_SPLIT.csv',rows);write_csv(out/'WITHIN_METRIC_CHOICES.csv',selections)
 write_csv(out/'WITHIN_METRIC_SUMMARY.csv',summary);write_csv(out/'FIXED_EXTREMA_SUMMARY.csv',extrema)
 write_csv(out/'ORIGINAL_WINNER_METRICS.csv',counts)
 write_json(ROOT/'audit/D3_AUDIT.json',dict(status='COMPLETE_AUTHOR_REEXTRACTION',candidate_records=len(rows),record_field_checks=field_checks,original_source_sha256=CONTRACT['sources']['s28_results']['sha256'],new_fitting=False))
 print('D3 complete',field_checks)
 for r in summary:
  if r['variant']=='ridge_1e4':print(r['metric'],r['family'],r['success_percent_median'],r['event_loss_min_median'])
if __name__=='__main__':main()
