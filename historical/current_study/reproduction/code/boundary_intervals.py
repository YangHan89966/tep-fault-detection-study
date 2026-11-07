"""Supplementary per-fault binomial intervals; original frozen bootstrap outputs retained."""
from common import *
from scipy.stats import beta,binomtest
import shutil

def main():
 src=ROOT/'Supporting_Data/v14_sources/S26_NEWSEED_PER_FAULT_ENDPOINTS_AND_INTERVALS.csv'
 rs=csv_rows(src.read_bytes());out=[]
 for row in rs:
  if row['endpoint'] not in ['detection_60','detection_full']:continue
  n=250;p=float(row['estimate']);x=int(round(n*p));assert abs(x/n-p)<1e-12
  low=0. if x==0 else float(beta.ppf(.025,x,n-x+1));high=1. if x==n else float(beta.ppf(.975,x+1,n-x))
  ci=binomtest(x,n).proportion_ci(.95,method='exact');assert abs(low-ci.low)<1e-10 and abs(high-ci.high)<1e-10
  out.append(dict(fault_id=int(row['fault_id']),method=row['method'],endpoint=row['endpoint'],successes=x,n=n,
   estimate_percent=100*p,CP_low_percent=100*low,CP_high_percent=100*high,
   original_bootstrap_low_percent=100*float(row['low']),original_bootstrap_high_percent=100*float(row['high']),
   inference_scope='additional per-fault fixed-state binomial model; independent Bernoulli runs assumed; not simultaneous or a pooled macro interval'))
 assert len(out)==136
 write_csv(ROOT/'results/INTERVALS/PER_FAULT_BINOMIAL_SUPPLEMENT.csv',out)
 write_json(ROOT/'audit/INTERVAL_AUDIT.json',dict(rows=len(out),source_sha256=sha(src),original_source_modified=False,
  checks='beta-quantile implementation agrees with independent binomtest inversion to 1e-10',
  all_success250_lower_percent=100*float(beta.ppf(.025,250,1)),macro_pooled_binomial_interval_created=False))
 print('Interval supplement',len(out),'250/250',100*float(beta.ppf(.025,250,1)))
if __name__=='__main__':main()
