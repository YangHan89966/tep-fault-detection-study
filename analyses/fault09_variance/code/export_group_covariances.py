"""Export the endpoint group moments from the saved sufficient statistics."""
from pathlib import Path
import csv
import numpy as np

root=Path(__file__).resolve().parents[1]
with np.load(root/'results/PER_RUN_SUFFICIENT_STATISTICS.npz',allow_pickle=False) as z:
    means=z['score_means']; cross=z['score_within_cross']; names=z['feature_names'].tolist()
d=means-means.mean(axis=0)
c=(cross.sum(axis=0)+960*d.T@d)/(400*960-1)
rows=[]
for a,b in [('C','C'),('A','A'),('U','U'),('C','A'),('C','U'),('A','U')]:
    factor=1 if a==b else 2
    vals=[]
    for prefix in ['rho0','post']:
        i,j=names.index(prefix+'_group_'+a),names.index(prefix+'_group_'+b)
        vals.append(float(factor*c[i,j]))
    rows.append({'component':('variance_'+a if a==b else 'twice_cov_'+a+'_'+b),
                 'rho0':vals[0],'post_update':vals[1],'post_minus_rho0':vals[1]-vals[0]})
with (root/'results/GROUP_ENDPOINT_COVARIANCES.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Exported six endpoint group moments.')
