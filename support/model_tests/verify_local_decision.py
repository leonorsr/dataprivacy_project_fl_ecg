from pathlib import Path
import sys,unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from support.local_decision import score
from support import final_candidates as final

class LocalDecision(unittest.TestCase):
    def data(self):
        a,b=final.CANDIDATES;rows=[];timings=[]
        for candidate in final.CANDIDATES:
            for seed in final.SEEDS:
                timings.append(dict(candidate=candidate,seed=seed,elapsed_seconds=10))
                for ds in final.DOMAINS:
                    for approach in ['local_teacher','global_student']:
                        value=(.8 if candidate==a else .6) if approach=='local_teacher' else (.3 if candidate==a else .9)
                        rows.append(dict(candidate=candidate,dataset=ds,seed=seed,approach=approach,split='dev',pr_auc_ap=value))
        return pd.DataFrame(rows),pd.DataFrame(timings)
    def test_teacher_winner_overrides_different_global_winner(self):
        rows,timings=self.data();self.assertEqual(score(rows,timings)[0],next(iter(final.CANDIDATES)))
    def test_all_local_seed_origin_pairs_required(self):
        rows,timings=self.data();rows=rows.drop(rows[rows.approach=='local_teacher'].index[0])
        with self.assertRaises(AssertionError):score(rows,timings)
    def test_holdout_cannot_replace_dev(self):
        rows,timings=self.data();rows.loc[rows.approach=='local_teacher','split']='train'
        with self.assertRaises(AssertionError):score(rows,timings)
if __name__=='__main__':unittest.main()
