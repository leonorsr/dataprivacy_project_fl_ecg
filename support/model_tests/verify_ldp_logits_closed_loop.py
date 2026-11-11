"""Budget accounting, client-side perturbation and real-metric closed-loop validation."""
import sys,tempfile,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import unittest,numpy as np,torch
from support.ldp_logits import closed_loop as exp
from support.ldp_logits.mechanism import ClientMechanism,Ledger
from support.model_tests.verify_global import domains

class ClosedLoop(unittest.TestCase):
 def test_secure_noise_and_three_release_composition(self):
  z=np.random.default_rng(6).normal(size=(32,2))
  with tempfile.TemporaryDirectory() as temp:
   for mechanism,delta in [('laplace',0.),('gaussiano',1e-5)]:
    c=dict(name='test',mode='ldp',mechanism=mechanism,representation='margem',clip_C=.5,epsilon_total=4.,delta_total=delta)
    ledger=Ledger(Path(temp)/(mechanism+'.json'));attempt=ledger.begin('test');m=ClientMechanism(c,3,ledger,attempt,'client')
    samples=[m.release(z) for _ in range(3)]
    self.assertFalse(np.array_equal(samples[0],samples[1]));self.assertTrue(m.certificate()['secure_randomness'])
    self.assertAlmostEqual(ledger.data['clients']['client']['epsilon_private_sum'],4.)
    self.assertAlmostEqual(ledger.data['clients']['client']['delta_private_sum'],delta)
    with self.assertRaises(RuntimeError):m.release(z)
    self.assertEqual(ledger.data['clients']['client']['private_releases'],3)

 def test_profiles_train_validate_and_have_real_feedback(self):
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True);data=domains()
  cfg=dict(exp.core.DESIGN,local_epochs=1,server_steps=1)
  profiles=[c for c in exp.conditions() if c['name'] in ['P1_clipped','P1_ldp','P3_ldp']]
  self.assertEqual(len(profiles),3)
  with tempfile.TemporaryDirectory() as temp:
   ledger=Ledger(Path(temp)/'ledger.json')
   for c in profiles:
    base=Path(temp)/c['name'];folder=base/'artifacts/2031';attempt=ledger.begin(c['name'])
    engine,mechanisms=exp.engine(c,ledger,attempt,folder)
    engine.run_one(data,'Softlabels',False,2031,settings=dict(cfg,ldp_condition=c),base=base)
    exp.core.validate_run(folder,data)
    self.assertEqual(len(mechanisms),4)
    self.assertTrue(all(m.releases==3 for m in mechanisms.values()))
    self.assertTrue(all(json.loads(p.read_text())['releases']==3 for p in (folder/'certificates').glob('*.json')))
    meta=json.loads((folder/'run.json').read_text());self.assertEqual(meta['condition'],c['name'])
   # The different private messages must actually change the trained global model.
   a=json.loads((Path(temp)/'P1_clipped/artifacts/2031/run.json').read_text())
   b=json.loads((Path(temp)/'P1_ldp/artifacts/2031/run.json').read_text())
   self.assertNotEqual(a['final_global_hash'],b['final_global_hash'])

if __name__=='__main__':unittest.main()
