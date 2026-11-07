"""Recalculate four S28 summaries from the supplied 8,400 selected-system rows."""
from pathlib import Path
import tempfile, sys, json
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from summarize import summarize_run

def main():
    names=['ALL_METHODS_SUMMARY.csv','ALL_PAIRED_DIFFERENCES.csv','PAIRED_SUMMARY.csv','PRIMARY_EQUAL_BUDGET.csv']
    with tempfile.TemporaryDirectory() as tmp:
        count=summarize_run(ROOT,Path(tmp)/'reports')
        checked={}
        for name in names:
            actual=pd.read_csv(Path(tmp)/'reports'/name)
            expected=pd.read_csv(ROOT/'reports'/name)
            pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-12,atol=1e-10)
            checked[name]={'rows':len(actual),'columns':len(actual.columns)}
    print(json.dumps({'status':'PASS','selected_system_rows':count,'checked':checked,'candidate_model_checks':False},indent=2))
if __name__=='__main__':main()
