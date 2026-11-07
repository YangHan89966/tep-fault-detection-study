"""Locate and read the two completed panels without modifying them.
Preflight inspects names/metadata only. Evaluation numerical reads require a lock.
"""
from __future__ import annotations
import io, json, os
from pathlib import Path, PurePosixPath
import numpy as np
from common import ROOT,sha,read_json,finite

def winpath(value):
    s=str(value).strip().strip('"')
    if len(s)>2 and s[1]==':' and s[0].isalpha():return Path('/mnt/'+s[0].lower()+'/'+s[2:].lstrip('\\/').replace('\\','/'))
    return Path(s).expanduser()
def candidates_from(base):
    base=winpath(base)
    if not base.is_dir():return []
    out=[];queue=[(base,0)];seen=set()
    while queue:
        p,d=queue.pop(0)
        if p.resolve() in seen:continue
        seen.add(p.resolve())
        for cp in (p,p/'workspace'):
            if (cp/'cases').is_dir() and (cp/'STATE_LOCK.json').is_file():out.append(cp)
        if d>=4:continue
        try:ds=list(p.iterdir())
        except OSError:continue
        for q in ds:
            if q.is_dir() and not q.is_symlink() and (q.name.startswith('TEP_F9_') or q.name in ('workspace','Desktop','Downloads')):queue.append((q,d+1))
    return out

def locate(manifest,roots,config=None):
    config=read_json(ROOT/'SOURCE_PATHS.json') if config is None else config
    choices=[]
    for p in roots:choices.extend(candidates_from(p))
    found={};report=[]
    for panel,info in manifest['panels'].items():
        supplied=config.get(panel,'').strip();scanned=candidates_from(supplied) if supplied else choices
        good=[];seen=set()
        for ws in scanned:
            if ws.resolve() in seen:continue
            seen.add(ws.resolve())
            try:
                if read_json(ws/'STATE_LOCK.json').get('protocol_sha256')!=info['protocol_sha256']:continue
            except (OSError,ValueError):continue
            missing=[c['case_id'] for c in info['cases'] if not (ws/'cases'/c['case_id']/'COMPLETE.json').is_file() or not (ws/'cases'/c['case_id']/c['raw_file']).is_file()]
            report.append(dict(panel=panel,workspace=str(ws),required=len(info['cases']),found=len(info['cases'])-len(missing),missing_cases=missing))
            if not missing:good.append(ws.resolve())
        if good:found[panel]=sorted(good,key=lambda p:(len(str(p)),str(p)))[0]
    return found,report

def raw_path(ws,c):
    rel=PurePosixPath(c['raw_file']);folder=Path(ws)/'cases'/c['case_id'];p=folder/c['raw_file']
    if rel.is_absolute() or '..' in rel.parts or p.is_symlink() or not p.resolve().is_relative_to(folder.resolve()):raise ValueError('Unsafe existing raw path')
    if sha(folder/'COMPLETE.json')!=c['complete_sha256']:raise ValueError('Original completion record changed: '+c['case_id'])
    return p

def check_views(raw,fault_id):
    n=finite(raw['native_values']);s=finite(raw['fast_control_input_measurements']);phase=raw['phase'];t=finite(raw['native_time_seconds']);idv=raw['diagnostic_IDV']
    if n.shape!=(960,52) or s.shape!=(960,41) or phase.shape!=(960,4):raise ValueError('Raw recording dimensions')
    if not np.all(phase[:,3]==3) or not np.array_equal(phase[:,0],np.arange(1,961)*180):raise ValueError('Raw recording phase')
    if t.shape!=(961,) or not np.allclose(t,np.arange(961)*180.,atol=1e-5,rtol=0):raise ValueError('Raw recording times')
    if idv.shape[0]!=960 or idv.shape[1]<9:raise ValueError('Fault flag dimensions')
    want=np.zeros_like(idv)
    if fault_id==9:want[160:,8]=1
    elif fault_id!=0:raise ValueError('Only normal/Fault9 allowed')
    if not np.array_equal(idv,want):raise ValueError('Fault injection mismatch')
    alt=n.copy();alt[:,:41]=s
    return {'native':n,'snapshot':alt}

def load_case(ws,c,evaluation_permitted=False):
    if c['role']!='normal_calibration' and not evaluation_permitted:raise ValueError('Evaluation numerical read before ALL threshold locks')
    p=raw_path(ws,c)
    if sha(p)!=c['raw_sha256']:raise ValueError('Changed original raw array: '+str(p))
    with np.load(p,allow_pickle=False) as a:
        raw={k:a[k].copy() for k in ('native_values','fast_control_input_measurements','phase','native_time_seconds','diagnostic_IDV')}
    return check_views(raw,c['fault_id'])

def load_raw_role(panel,role,manifest,locations,evaluation_permitted=False):
    cases=[c for c in manifest['panels'][panel]['cases'] if c['role']==role]
    v={'native':[],'snapshot':[]}
    for i,c in enumerate(cases,1):
        views=load_case(locations[panel],c,evaluation_permitted)
        for key in v:v[key].append(views[key])
        if i%25==0 or i==len(cases):print(f'READ EXISTING {panel}/{role} {i}/{len(cases)} | NO SIMULATION',flush=True)
    return {k:np.stack(a) for k,a in v.items()},cases
