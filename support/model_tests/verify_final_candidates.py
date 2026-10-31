"""Decision-rule checks; no expensive MobileNet training."""
from pathlib import Path
import sys,unittest
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from support.final_candidates import CANDIDATES,SEEDS,DOMAINS,decide

class DecisionChecks(unittest.TestCase):
    def inputs(self,values):
        rows=[];timings=[]
        for candidate in CANDIDATES:
            for seed in SEEDS:
                timings.append(dict(candidate=candidate,seed=seed,elapsed_seconds=20 if candidate.startswith('A') else 10))
                for ds in DOMAINS:rows.append(dict(candidate=candidate,dataset=ds,seed=seed,split='dev',approach='global_student',pr_auc_ap=values[candidate][ds]))
        return pd.DataFrame(rows),pd.DataFrame(timings)
    def test_smaller_worst_origin_loss_wins(self):
        a,b=CANDIDATES
        x,t=self.inputs({a:dict(zip(DOMAINS,[.7,.5])),b:dict(zip(DOMAINS,[.5,.6]))})
        winner,mean,ranking,paired,differences=decide(x,t)
        self.assertEqual(winner,a)
        self.assertAlmostEqual(ranking.loc[a,'worst_origin_regret'],.1)
        self.assertAlmostEqual(ranking.loc[b,'worst_origin_regret'],.2)
    def test_exact_metric_tie_uses_runtime(self):
        a,b=CANDIDATES;x,t=self.inputs({a:dict(zip(DOMAINS,[.6,.6])),b:dict(zip(DOMAINS,[.6,.6]))})
        self.assertEqual(decide(x,t)[0],b)
    def test_incomplete_seeds_rejected(self):
        a,b=CANDIDATES;x,t=self.inputs({a:dict(zip(DOMAINS,[.6,.6])),b:dict(zip(DOMAINS,[.6,.6]))})
        with self.assertRaises(AssertionError):decide(x[x.seed!=SEEDS[-1]],t)

    def test_local_epoch_budget_and_global_distribution(self):
        import tempfile,torch
        from collections import defaultdict
        from support import final_candidates as final
        from support.model_tests.verify_global import domains
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        engine=final.training_engine();epoch_counts=defaultdict(int)
        original_fit=engine.learning.fit
        def counted_fit(model,name,client,epochs,seed,**kwargs):
            epoch_counts[id(model)]+=epochs
            return original_fit(model,name,client,epochs,seed,**kwargs)
        try:
            with tempfile.TemporaryDirectory() as temp:
                engine.SOFT=Path(temp);engine.learning.fit=counted_fit
                data=domains();cfg=final.configurations()[0]
                engine.run_pair(cfg,data)
                self.assertEqual(sorted(epoch_counts.values()),[30]*4+[32]*4)
                validator=final.isolate('_final_epoch_budget_validator','validation.py')
                validator.SOFT=Path(temp);validator.signature=final.signature;validator.soft_domains=lambda _:data
                report=validator.validate_soft(require_complete=False)
                self.assertTrue(report['trained_server_student_verified'])
                self.assertTrue(report['redistributed_checkpoints_verified'])
        finally:engine.learning.fit=original_fit

if __name__=='__main__':unittest.main()
