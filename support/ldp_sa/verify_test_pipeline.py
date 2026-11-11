"""Exercise the entire final-test evaluator with synthetic shards, never real test."""
import hashlib,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from . import experiment as exp
from . import test_evaluation as evaluation
from . import notebook
from support.federated_global.shared import dump,model_hash
from support.ldp_logits import training_base as core

def verify():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix='ldp-sa-test-evaluator-') as directory:
        base=Path(directory);domains={};roots={}
        for ds in ['icentia11k','physionet2017']:
            root=base/ds;root.mkdir();roots[ds]=root
            shard=root/'synthetic.npz';rng=np.random.default_rng(17)
            np.savez_compressed(shard,X=rng.normal(size=(4,500)).astype('float32'),
                features=rng.normal(size=(4,2)).astype('float32'),y=np.array([0,1,2,3]),
                sample_ids=np.array(['test_'+str(i) for i in range(4)]),valid_lengths=np.full(4,500))
            pd.DataFrame([dict(split='test',client_id='client_99',shard_path=shard.name,
                sha256=hashlib.sha256(shard.read_bytes()).hexdigest())]).to_csv(root/'manifest.csv',index=False)
            domains[ds]=dict(config={'feature_names':['a','b']},clients={'client_00':dict(split='train',
                features=np.array([[1.,2.],[2.,4.],[3.,5.]]),fit_indices=np.array([0,1,2]),sample_ids=np.array(['fit_a','fit_b','fit_c']))})
        with patch.object(exp,'BASE',base),patch.object(evaluation,'BASE',base),patch.object(evaluation,'ROOTS',roots),patch.object(notebook,'refresh',lambda:None):
            token=exp.signature()
            for seed in exp.DESIGN['seeds']:
                folder=exp.folder_for(seed);folder.mkdir(parents=True)
                model=core.soft.GlobalCNN1D({ds:2 for ds in domains},seed)
                torch.save(model.state_dict(),folder/'global_round_3.pt')
                dump(folder/'run.json',dict(final_global_hash=model_hash(model)))
            dump(base/'test_gate.json',dict(signature=token,training_complete=True,selection_on_test=False))
            evaluation.evaluate_test(domains)
            report=json.loads((base/'test/validation.json').read_text())
            assert report['n_examples_once']==6 and report['seed_count']==5
            rows=json.loads((base/'test/metrics.json').read_text())
            assert len(rows)==25 and all(r['role']=='global' for r in rows)
            assert all(r['n_examples']==3 for r in rows if r['scope']=='client')
            assert report['test_metrics_recomputed'] and not report['selection_on_test']
    return True

if __name__=='__main__':
    print('Synthetic full-test evaluator:',verify())
