"""Signed secure FedAvg and actual paired training on tiny, unequal-fit clients."""
import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from support import sa_communication as exp
from support.model_tests.verify_global import domains

class Communication(unittest.TestCase):
    def test_signed_secure_sum_matches_weighted_fedavg(self):
        x=np.array([[-2.,1.,4.],[3.,-4.,2.],[1.,2.,-3.]])
        actual,audit=exp.aggregate(x,[2,4,6],True,'Fedavg',bits=24,threshold=2,bound=8.)
        expected=np.average(x,axis=0,weights=[2,4,6])
        self.assertTrue(np.allclose(actual,expected,atol=3*16/(2*2**24)))
        self.assertTrue(audit['secure_integer_sum_verified'])
        self.assertGreater(audit['upload_bytes'],audit['baseline_upload_bytes'])
        with self.assertRaises(AssertionError):exp.aggregate(x,[2,4,6],True,'Fedavg',threshold=2,bound=1.)

    def test_four_conditions_train_validate_and_detect_corruption(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        data=domains();data['physionet2017']['clients']['client_00']['fit_indices']=np.arange(2)
        cfg=dict(exp.DESIGN,rounds=2,local_epochs=1,server_steps=1,threshold=3)
        with tempfile.TemporaryDirectory() as temp:
            for kind in ['Fedavg','Softlabels']:
                for sa in [False,True]:
                    exp.run_one(data,kind,sa,2031,settings=cfg,base=temp)
                    folder=Path(temp)/kind/'artifacts/2031'/('SA' if sa else 'plain')
                    meta=json.loads((folder/'run.json').read_text())
                    self.assertEqual(meta['status'],'complete');self.assertAlmostEqual(meta['source_weights']['icentia11k'],.6)
                    exp.validate_run(folder,data)
                # Round-one local training must be paired exactly: SA acts only afterwards.
                a=Path(temp)/kind/'artifacts/2031/plain';b=a.parent/'SA'
                import pandas as pd
                left=pd.read_json(a/'metrics.json');right=pd.read_json(b/'metrics.json')
                for l,r in zip(left[(left['round']==1)&(left.role=='local')].to_dict('records'),right[(right['round']==1)&(right.role=='local')].to_dict('records')):
                    with np.load(a/'predictions'/l['prediction_file']) as x,np.load(b/'predictions'/r['prediction_file']) as y:
                        self.assertTrue(np.array_equal(x['p'],y['p']))
            # Recompute from predictions: a corrupted saved metric must be rejected.
            folder=Path(temp)/'Fedavg/artifacts/2031/SA';path=folder/'metrics.json';rows=json.loads(path.read_text())
            rows[0]['pr_auc_ap']=-1.;path.write_text(json.dumps(rows))
            with self.assertRaises(AssertionError):exp.validate_run(folder,data)

if __name__=='__main__':unittest.main()
