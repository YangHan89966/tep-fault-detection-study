"""Check S28 release hashes and structural coverage without numerical fitting."""
from pathlib import Path, PurePosixPath
import argparse, collections, csv, hashlib, io, json, zipfile

def verify(read, names):
    lines=read("RESULTS_SHA256.txt").decode().splitlines()
    expected={}
    for line in lines:
        value,name=line.split("  ",1)
        p=PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or name in expected or name=="RESULTS_SHA256.txt":
            raise ValueError("Unsafe or duplicate manifest member: "+name)
        expected[name]=value
    if set(expected)!=(set(names)-{"RESULTS_SHA256.txt"}):
        raise ValueError("File inventory differs from the release manifest")
    for name,value in expected.items():
        if hashlib.sha256(read(name)).hexdigest()!=value:
            raise ValueError("SHA-256 mismatch: "+name)
    rows=list(csv.DictReader(io.StringIO(read("MAIN_RESULTS.csv").decode("utf-8-sig"))))
    groups=collections.Counter((int(r["fault"]),r["split_id"],r["variant"]) for r in rows)
    wanted={(f,"D%02d"%d,v) for f in range(1,21) for d in range(1,11) for v in ("baseline","ridge_1e4","truncate_1e4")}
    if len(rows)!=8400 or set(groups)!=wanted or set(groups.values())!={14}:
        raise ValueError("Completed S28 result coverage differs")
    print(json.dumps({"sha256_files_checked":len(expected),"faults":20,"branches":600,"result_rows":8400,"status":"PASS_FILE_INTEGRITY_AND_COVERAGE"},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument("--root",type=Path);g.add_argument("--archive",type=Path)
    a=p.parse_args()
    if a.archive:
        with zipfile.ZipFile(a.archive) as z:
            names=[n for n in z.namelist() if not n.endswith("/")]
            if len(names)!=len(set(names)):raise ValueError("Duplicate archive names")
            verify(z.read,names)
    else:
        root=a.root.resolve()
        names=[p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        def read(name):
            p=root/name
            if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError("Unsafe file path")
            return p.read_bytes()
        verify(read,names)
if __name__=="__main__":main()
