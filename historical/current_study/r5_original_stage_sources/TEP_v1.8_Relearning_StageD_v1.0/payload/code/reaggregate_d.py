"""Reaggregate a Stage D return from saved alarm bits only, into a NEW directory.
Run: python -B reproduction_code/reaggregate_d.py --return-root EXTRACTED_RETURN --output NEW_DIRECTORY
No simulator, raw trajectories, fitting or calibration is invoked.
"""
from pathlib import Path
import argparse,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent))
from storage import strict_json,verify_tree
from analysis_d import aggregate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--return-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new empty destination name; nothing overwritten')
    verify_tree(a.return_root)
    spec=strict_json(a.return_root/'reproduction_spec.json');protocol=strict_json(a.return_root/'reproduction_reference/EXPERIMENT_1_RELEARNING_PROTOCOL_v1.0.json')
    aggregate(a.return_root/'evaluated',a.output,spec,protocol['primary_contrasts'])
    print('Saved-mask reaggregation complete. This did not score raw inputs or repeat learning.')
if __name__=='__main__':main()
