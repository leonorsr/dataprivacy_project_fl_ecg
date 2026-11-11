"""Tiny integration proof: trained server checkpoint, exact broadcast and KD back to teachers."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from support.federated_global import shared,soft,validation


def domains():
    rng=np.random.default_rng(19);out={}
    for ds,nfeatures in [('icentia11k',3),('physionet2017',5)]:
        clients={}
        for j in range(3):
            n=8 if j<2 else 2;cid=f'client_{j:02}';y=np.arange(n)%2
            c=dict(uid=ds+':'+cid,dataset=ds,client_id=cid,split='train' if j<2 else 'dev',
                X=rng.normal(size=(n,256)).astype(np.float32),F=rng.normal(size=(n,nfeatures)).astype(np.float32),
                Z=rng.normal(size=(n,32+nfeatures)).astype(np.float32),y=y.astype(np.int64),
                valid_lengths=np.full(n,256,dtype=np.int64),sample_ids=np.array([f'{ds}:{j}:{i}' for i in range(n)]),
                fit_indices=np.arange(6) if j<2 else np.array([],int),hold_indices=np.arange(6,8) if j<2 else np.arange(2))
            clients[cid]=c
        anchor=dict(X=rng.normal(size=(32,256)).astype(np.float32),F=rng.normal(size=(32,nfeatures)).astype(np.float32),
            Z=rng.normal(size=(32,32+nfeatures)).astype(np.float32),valid_lengths=np.full(32,256,dtype=np.int64),sample_ids=np.array([f'synthetic_{i}' for i in range(32)]))
        out[ds]=dict(clients=clients,anchor=anchor,config={'feature_names':[f'f{i}' for i in range(nfeatures)]})
    return out

class GlobalFlow(unittest.TestCase):
    def test_heterogeneous_teachers_and_student_broadcast(self):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        old_soft,old_validation,loader=soft.SOFT,validation.SOFT,validation.soft_domains
        try:
            with tempfile.TemporaryDirectory() as temp:
                soft.SOFT=validation.SOFT=Path(temp);data=domains();validation.soft_domains=lambda _:data
                cfg=dict(experiment='pilot',name='tiny',models={'icentia11k':'CNN1D','physionet2017':'CNN_BiLSTM'},assignments={},seed=2026)
                soft.run_pair(cfg,data)
                report=validation.validate_soft(require_complete=False)
                self.assertTrue(report['trained_server_student_verified'])
                self.assertEqual(report['validated_pairs'],1)
                folder=Path(temp)/'pilot/tiny/2026';meta=json.loads((folder/'run_config.json').read_text())
                self.assertEqual(len(meta['final_recipients']),4)
                before=(folder/'global_final.pt').read_bytes();soft.run_pair(cfg,data)
                self.assertEqual(before,(folder/'global_final.pt').read_bytes())
                # Invalid broadcast must be detected even if metrics themselves look correct.
                receipt=next((folder/'receipts').glob('*.json'));rows=json.loads(receipt.read_text());rows[-1]['received_global_hash']='invalid';receipt.write_text(json.dumps(rows))
                with self.assertRaises(AssertionError):validation.validate_soft(require_complete=False)
        finally:soft.SOFT=old_soft;validation.SOFT=old_validation;validation.soft_domains=loader
    def test_fedavg_aggregation_and_broadcast(self):
        from types import SimpleNamespace
        from flwr.common import FitRes,Status,Code,ndarrays_to_parameters,parameters_to_ndarrays
        from support.federated_global.fedavg import AuditedFedAvg
        with tempfile.TemporaryDirectory() as temp:
            strategy=AuditedFedAvg(['a','b'],['dev'],[np.zeros(2,dtype=np.float32),np.array(0,dtype=np.int64)],Path(temp))
            manager=SimpleNamespace(all=lambda:{k:SimpleNamespace(cid=k) for k in ['a','b','dev']})
            params=ndarrays_to_parameters([np.zeros(2,dtype=np.float32),np.array(0,dtype=np.int64)])
            self.assertEqual({c.cid for c,_ in strategy.configure_fit(1,params,manager)},{'a','b'})
            responses=[]
            for uid,weight,v,counter in [('a',1,[1.,3.],1),('b',3,[5.,7.],2)]:
                arrays=[np.array(v,dtype=np.float32),np.array(counter,dtype=np.int64)]
                responses.append((SimpleNamespace(cid=uid),FitRes(Status(Code.OK,''),ndarrays_to_parameters(arrays),weight,{})))
            result,_=strategy.aggregate_fit(1,responses,[])
            arrays=parameters_to_ndarrays(result)
            self.assertTrue(np.allclose(arrays[0],[4.,6.]))
            self.assertEqual(arrays[1].dtype,np.dtype('int64'))
            self.assertEqual({c.cid for c,_ in strategy.configure_evaluate(1,result,manager)},{'a','b','dev'})
            strategy.configure_fit(2,result,manager)

    def test_supported_candidates(self):
        self.assertEqual(len(shared.ELIGIBLE),10)
        self.assertFalse(set(shared.ELIGIBLE)&{'RandomForest','KNN','GRU5','GRU_Simple'})

if __name__=='__main__':unittest.main()
