#!/usr/bin/env python3
"""Verify a fixed D2/D3 input; optionally run preserved code in a new output tree.

Public archive derivatives remove host paths only. This adapter changes only the
copied contract's input path/hash, while preserving all numerical code and settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--diagnostic',choices=('D2','D3'),required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--run',action='store_true',help='Run original diagnostic after checking the fixed input; default only verifies input')
    ap.add_argument('--out',type=Path,help='New output directory outside the repository; required with --run')
    a=ap.parse_args()
    role='s30_paths' if a.diagnostic=='D2' else 's28_results'
    index=json.loads((REPO/'current_study/DIAGNOSTIC_INPUT_CATALOG.json').read_text())
    entry=next(x for x in index['inputs'] if x['role']==role)
    source=a.source.resolve(strict=True)
    actual=sha(source)
    choices={entry['original_sha256']:'original',entry['public_sha256']:'prior_public_path_redacted_derivative'}
    if entry.get('current_distribution'):
        choices[entry['current_distribution']['sha256']]='current_distribution'
    if actual not in choices:raise ValueError('Input archive SHA-256 differs from the recorded source or distribution identities')
    report={'diagnostic':a.diagnostic,'input_kind':choices[actual],
            'supplied_archive_sha256':actual,'original_archive_sha256':entry['original_sha256'],
            'scientific_code_modified':False,'simulation':False,'new_fitting':False}
    if not a.run:
        report['status']='INPUT_IDENTITY_VERIFIED_NOT_EXECUTED'
        print(json.dumps(report,indent=2));return 0
    if a.out is None:ap.error('--out is required with --run')
    out=a.out.absolute()
    if out.exists():raise ValueError('A new output directory is required')
    resolved=out.resolve()
    if resolved==REPO or REPO in resolved.parents or resolved==source.parent:
        raise ValueError('Output must be outside the repository and separate from the input file directory')
    script='d2_order_threshold.py' if a.diagnostic=='D2' else 'd3_existing_metrics.py'
    original=REPO/'current_study/reproduction'
    for name in ('common.py',script):
        wanted=index['preserved_code_sha256'][name]
        if sha(original/'code'/name)!=wanted:raise ValueError('Preserved scientific code identity mismatch: '+name)
    contract_path=original/'audit/ANALYSIS_CONTRACT.json'
    if sha(contract_path)!=index['analysis_contract_sha256']:
        raise ValueError('Distribution analysis contract identity mismatch')
    contract=json.loads(contract_path.read_text())
    if contract['sources'][role]['sha256']!=entry['original_sha256']:
        raise ValueError('Original contract/source binding mismatch')
    out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir();(out/'audit').mkdir()
    for name in ('common.py',script):shutil.copyfile(original/'code'/name,out/'code'/name)
    shutil.copyfile(contract_path,out/'audit/ANALYSIS_CONTRACT.original.json')
    contract['sources'][role]['path']=str(source)
    contract['sources'][role]['sha256']=actual
    (out/'audit/ANALYSIS_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
    report.update({'status':'STARTED','contract_changes':[
        'sources.'+role+'.path','sources.'+role+'.sha256'],
        'note':'Only transport location/hash changed in the working copy. Original diagnostic audit fields referring to the supplied source hash describe this checked input; original ZIP identity is recorded separately here.'})
    report_path=out/'audit/PUBLICATION_INPUT_CONTEXT.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    cmd=[sys.executable,'-B',str(out/'code'/script),'--source',str(source)]
    if a.diagnostic=='D2':cmd+=['--out',str(out/'results/D2')]
    result=subprocess.run(cmd,cwd=out)
    report['exit_code']=result.returncode
    report['status']='COMPLETED' if result.returncode==0 else 'FAILED'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return result.returncode

if __name__=='__main__':
    raise SystemExit(main())
