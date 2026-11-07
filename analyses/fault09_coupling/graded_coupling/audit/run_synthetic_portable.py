"""Run the synthetic checks from a portable public-package location.

The public test has one documented path-only derivation from the historical
file. This wrapper redirects ROOT and OUT in memory and leaves that public
source unchanged. Output is written to a separate caller-selected directory.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,
                        help='Unpacked graded-coupling package containing CODE_LOCK.json')
    parser.add_argument('--output',type=Path,
                        help='Separate directory for the new synthetic-only review JSON')
    args=parser.parse_args()
    root=args.root.expanduser().resolve(strict=True)
    output=(args.output.expanduser().resolve() if args.output else
            root.parent/(root.name+'_synthetic_recheck'))
    if output==root or output.is_relative_to(root):
        raise ValueError('Use a separate output directory outside the frozen package')
    script=root/'audit'/'test_coupling_synthetic.py'
    lock=json.loads((root/'CODE_LOCK.json').read_text(encoding='utf-8'))
    public_spec=importlib.util.spec_from_file_location('_public_coupling_identity',root/'public_identity.py')
    public_identity=importlib.util.module_from_spec(public_spec)
    public_spec.loader.exec_module(public_identity)
    public_identity.verify_frozen_files(root)
    original_hash=file_hash(script)
    tree=ast.parse(script.read_text(encoding='utf-8'),filename=str(script))
    changed=[]
    for statement in tree.body:
        if (isinstance(statement,ast.Assign) and len(statement.targets)==1
                and isinstance(statement.targets[0],ast.Name)
                and statement.targets[0].id in ('ROOT','OUT')):
            name=statement.targets[0].id
            value=root if name=='ROOT' else output
            statement.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),
                                     args=[ast.Constant(str(value))],keywords=[])
            changed.append(name)
    if sorted(changed)!=['OUT','ROOT']:
        raise ValueError('Frozen test path assignments differ from expected structure')
    output.mkdir(parents=True,exist_ok=True)
    namespace={'__name__':'__main__','__file__':str(script)}
    exec(compile(ast.fix_missing_locations(tree),str(script),'exec'),namespace)
    if file_hash(script)!=original_hash:
        raise ValueError('Frozen test changed during execution')
    record={
        'status':'PASS_PORTABLE_SYNTHETIC_WRAPPER',
        'root':'.','output':'separate caller-selected directory',
        'public_test_sha256':original_hash,
        'original_frozen_test_sha256':lock['files']['audit/test_coupling_synthetic.py'],
        'in_memory_path_assignments_only':['ROOT','OUT'],
        'public_test_file_unchanged':True,
        'scientific_score_outputs_read':False,
        'numerical_report':'SYNTHETIC_NUMERICAL_REVIEW.json',
    }
    (output/'PORTABLE_WRAPPER_REVIEW.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Synthetic-only reports saved to: '+str(output),flush=True)


if __name__=='__main__':
    main()
