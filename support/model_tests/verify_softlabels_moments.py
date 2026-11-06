"""MC dropout preserves class association, RNG/BatchNorm, and validates actual training."""
import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from support import softlabels_moments as exp
from support.model_tests.verify_global import domains

class Moments(unittest.TestCase):
    def test_binary_reconstruction_and_class_identity(self):
        a=np.linspace(.1,.9,32);p=np.column_stack([1-a,a]);q=p[::-1]
        encoded=exp.encode(p,'anomaly32');self.assertTrue(np.allclose(np.column_stack([1-encoded,encoded]),p))
        left=exp.encode(p,'mc_meanstd64',np.full(32,.05)).reshape(32,2)
        right=exp.encode(q,'mc_meanstd64',np.full(32,.05)).reshape(32,2)
        self.assertFalse(np.array_equal(left,right));self.assertTrue(np.allclose(left[:,0],p[:,1]))

    def test_per_anchor_mixture_variance_includes_between_clients(self):
        a=np.column_stack([np.ones(32)*.9,np.ones(32)*.1]);b=a[:,::-1]
        result,comm=exp.aggregate([exp.encode(a,'mc_meanstd64',np.zeros(32)),exp.encode(b,'mc_meanstd64',np.zeros(32))],[1,3],'mc_meanstd64')
        result=result.reshape(32,2)
        self.assertTrue(np.allclose(result[:,0],.7));self.assertTrue(np.allclose(result[:,1]**2,.12))
        self.assertEqual(comm['values_per_client'],64);self.assertEqual(comm['upload_bytes'],2*(256+4))

    def test_mc_changes_predictions_but_not_rng_or_model_state(self):
        torch.set_num_threads(2);data=domains();ds='icentia11k';anchor=data[ds]['anchor']
        model=exp.core.learning.factory('CNN1D',3,256,2031);model.train()
        digest=exp.core.model_hash(model);rng=torch.get_rng_state().clone()
        p,std=exp.mc_predictions(model,'CNN1D',ds,anchor,seed=77,passes=20)
        self.assertEqual(digest,exp.core.model_hash(model));self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue((std>0).any())
        p2,std2=exp.mc_predictions(model,'CNN1D',ds,anchor,seed=77,passes=20)
        self.assertTrue(np.array_equal(p,p2));self.assertTrue(np.array_equal(std,std2))
        self.assertTrue(np.allclose(p.sum(1),1))

    def test_each_variant_trains_and_validates(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        data=domains();cfg=dict(exp.core.DESIGN,rounds=2,local_epochs=1,server_steps=1,threshold=3)
        with tempfile.TemporaryDirectory() as temp:
            for mode in exp.MODES:
                root=Path(temp)/mode;engine=exp.engine(mode,root=root)
                engine.run_one(data,'Softlabels',False,2031,settings=dict(cfg,representation=mode),base=root)
                folder=root/'artifacts/2031';exp.core.validate_run(folder,data)
                import pandas as pd
                c=pd.read_json(folder/'communication.json');expected=64 if mode in ['probabilities64','mc_meanstd64'] else 32
                self.assertEqual(set(c.values_per_client),{expected})
                with np.load(folder/'anchor_round_1.npz') as z:self.assertEqual(z['messages'].shape,(4,expected))

if __name__=='__main__':unittest.main()
