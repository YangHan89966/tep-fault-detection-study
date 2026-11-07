from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from dev_reader import read_dev_pickle,read_f18_prefix
from data_io import project
from old_evidence import load_npz
from worker import execute,ROOT
import argparse
a=argparse.ArgumentParser()
a.add_argument('--data-root',type=Path,required=True)
a.add_argument('--out',type=Path,required=True)
args=a.parse_args()
pca=load_npz(ROOT/"inputs/PCA_STATE.npz");basis=load_npz(ROOT/"inputs/RESIDUAL_BASIS.npz")["basis"]
n,ni=read_dev_pickle(args.data_root/'TEP_F18_OLD_NORMAL_DATA_20260905T055940_352230Z.zip','normal')
normal,err=project(n,pca,basis);del n
access={"mode":"LOCAL_OLD_F3_F18_ONLY","normal":ni,"faults":{},"old_final_process_rows_decoded":0,"new_seed_access":False}
def provider(fault):
    if fault==3:
        x,r=read_dev_pickle(args.data_root/'TEP_F3_OLD_DATA_20260905T064503_375561Z.zip','fault3')
    elif fault==18:
        x,r=read_f18_prefix(args.data_root/'TEP_F18_OLD_DATA_20260905T044916_102708Z.zip')
    else:raise ValueError("Local mode supports F3 and F18 only")
    obj,e=project(x,pca,basis);access["faults"][str(fault)]=r
    return obj
execute(ROOT,args.out,normal,provider,[3,18],access,"LOCAL_F3_F18_NOT_ALL20")
