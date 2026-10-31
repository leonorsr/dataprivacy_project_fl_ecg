"""Mechanism/accounting tests plus a tiny client->server->client LDP cycle."""
from pathlib import Path
import sys,tempfile,json,unittest
from collections import defaultdict
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from support.ldp.mechanism import clip_logits,LocalLogitMechanism,PrivacyLedger
from support.ldp import experiment

class LDPChecks(unittest.TestCase):
    def test_joint_clipping_and_privacy_loss_bound(self):
        rng=np.random.default_rng(5);bound=32.;epsilon=4
        a=clip_logits(rng.normal(size=(32,2))*100,bound)
        b=clip_logits(rng.normal(size=(32,2))*100,bound)
        self.assertLessEqual(np.abs(a).sum(),bound+1e-9)
        self.assertLessEqual(np.abs(a-b).sum(),2*bound+1e-9)
        scale=2*bound/(epsilon/3)
        for _ in range(20):
            y=rng.normal(size=(32,2))*100
            privacy_loss=(np.abs(y-b).sum()-np.abs(y-a).sum())/scale
            self.assertLessEqual(abs(privacy_loss),epsilon/3+1e-9)
    def test_centering_invariant_and_invalid_inputs(self):
        z=np.tile([.1,.2],(32,1))
        a=torch.tensor(z).softmax(-1).numpy();b=torch.tensor(clip_logits(z,32)).softmax(-1).numpy()
        self.assertTrue(np.allclose(a,b))
        self.assertTrue(np.array_equal(clip_logits(np.zeros((32,2))),np.zeros((32,2))))
        for epsilon in [0,-1,np.inf,np.nan]:
            with self.assertRaises(ValueError):LocalLogitMechanism('ldp',epsilon=epsilon)
        with self.assertRaises(ValueError):LocalLogitMechanism('ldp',epsilon=1,bound=0)
        with self.assertRaises(ValueError):clip_logits(np.full((32,2),np.nan))

    def test_joint_scale_and_budget_exhaustion(self):
        m=LocalLogitMechanism('ldp',epsilon=4,bound=32)
        self.assertEqual(m.noise_scale,48.)
        for _ in range(3):self.assertEqual(m.release(np.zeros((32,2))).shape,(32,2))
        with self.assertRaises(RuntimeError):m.release(np.zeros((32,2)))
        cert=m.certificate();self.assertEqual(cert['releases'],3)
        self.assertFalse(any(k in cert for k in ['raw_norm','noise_seed','clip_factor','unclipped_logits']))
    def test_public_training_seed_cannot_reproduce_private_noise(self):
        np.random.seed(7);a=LocalLogitMechanism('ldp',epsilon=1).release(np.zeros((32,2)))
        np.random.seed(7);b=LocalLogitMechanism('ldp',epsilon=1).release(np.zeros((32,2)))
        self.assertFalse(np.array_equal(a,b))
    def test_ledger_composes_attempts_and_marks_nonprivate_baselines(self):
        with tempfile.TemporaryDirectory() as temp:
            ledger=PrivacyLedger(Path(temp)/'ledger.json')
            for trial in range(2):
                attempt=ledger.begin(str(trial),3)
                m=LocalLogitMechanism('ldp',epsilon=3,ledger=ledger,attempt=attempt,client='a')
                m.release(np.zeros((32,2)))
            self.assertEqual(ledger.state['clients']['a']['epsilon_spent_private_releases'],2.)
            attempt=ledger.begin('baseline',None);m=LocalLogitMechanism('raw',ledger=ledger,attempt=attempt,client='a');m.release(np.zeros((32,2)))
            self.assertTrue(ledger.state['clients']['a']['non_private_release'])
    def test_closed_loop_student_receives_only_noisy_logits(self):
        from support.model_tests.verify_global import domains
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        previous=experiment.BASE
        try:
            with tempfile.TemporaryDirectory() as temp:
                experiment.BASE=Path(temp);data=domains();cfg=experiment.settings();cfg['public_fit_per_train_client']=6
                condition=dict(name='epsilon_4',mode='ldp',epsilon=4)
                ledger=PrivacyLedger(Path(temp)/'private_privacy_ledger.json');attempt=ledger.begin('unit',4)
                engine=experiment.build_engine(condition,ledger,attempt,cfg);engine.SOFT=Path(temp)
                run=dict(experiment='runs',name='epsilon_4',models={'icentia11k':'CNN1D','physionet2017':'CNN_BiLSTM'},assignments={},seed=2041,ldp_mode='ldp',epsilon_client_cycle=4,local_epochs_per_round=10)
                engine.run_pair(run,data)
                folder=Path(temp)/'runs/epsilon_4/2041'
                experiment.move_private_artifacts(folder);ledger.finish(attempt)
                result=experiment.validate(data,require_complete=False)
                self.assertTrue(result['ldp_message_mechanism_verified'])
                self.assertTrue(result['redistributed_checkpoints_verified'])
                for c in ledger.state['clients'].values():self.assertAlmostEqual(c['epsilon_spent_private_releases'],4.)
                experiment.export_server('epsilon_4',2041,Path(temp)/'server_export')
                self.assertTrue((Path(temp)/'server_export/global_final.pt').exists())
                self.assertFalse(list((Path(temp)/'server_export').glob('*prediction*')))
                self.assertFalse((Path(temp)/'server_export/metrics.json').exists())
        finally:experiment.BASE=previous

if __name__=='__main__':unittest.main()
