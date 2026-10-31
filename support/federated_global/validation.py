"""Recompute metrics and prove final checkpoints match distributed global replicas."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .shared import *
from .soft import GlobalCNN1D,predictions,configs
from global_fed.metrics import evaluate

def check_metrics(row,z,client):
    ix=client['hold_indices']
    assert np.array_equal(z['sample_ids'],client['sample_ids'][ix])
    assert np.array_equal(z['y'],client['y'][ix])
    assert np.isfinite(z['p']).all() and (z['p']>=0).all() and np.allclose(z['p'].sum(1),1)
    metrics,cm=evaluate(z['y'],z['p'],row['dataset']);assert np.array_equal(cm,z['cm'])
    for key,value in metrics.items():assert np.isclose(row[key],value,equal_nan=True,atol=1e-12),(key,row[key],value)

def validate_fedavg(require_complete=True):
    checked=0;completed=0
    for ds in DOMAINS:
        clients,cfg,_,_=raw.prepare(ds,80,64,2026,False)
        fit_ids={v for c in clients.values() for v in c['sample_ids'][c['fit_indices']]}
        for seed in SEEDS:
            for name in ELIGIBLE:
                folder=FED/f'{ds}_80_64_{seed}'/name;p=folder/'run_config.json'
                if not p.exists():continue
                config=json.loads(p.read_text())
                if config.get('status')!='complete' or config.get('protocol_signature')!=signature():continue
                assert not config['test_used'] and not config['test_clients']
                model=learning.factory(name,len(cfg['feature_names']),cfg['signal_vector_samples'],seed)
                audit=json.loads((folder/'aggregation_audit.json').read_text());assert len(audit)==3
                for i,a in enumerate(audit):
                    assert a['weighted_average_verified'] and a['global_model_created']
                    assert set(a['redistributed_to'])==set(clients)
                    if i:assert a['input_global_parameter_hash']==audit[i-1]['global_parameter_hash']
                model.load_state_dict(torch.load(folder/'global_final.pt',map_location='cpu',weights_only=True))
                assert model_hash(model)==audit[-1]['global_parameter_hash']
                for uid in config['train_clients']:
                    replica=learning.factory(name,len(cfg['feature_names']),cfg['signal_vector_samples'],seed)
                    replica.load_state_dict(torch.load(folder/'distributed'/f'{uid}.pt',map_location='cpu',weights_only=True))
                    assert model_hash(replica)==model_hash(model)
                frame=pd.read_json(folder/'metrics.json')
                for row in frame.to_dict('records'):
                    if not isinstance(row.get('prediction_file'),str):continue
                    client=clients[row['evaluation_client']]
                    with np.load(folder/row['prediction_file']) as z:
                        check_metrics(row,z,client);assert not fit_ids&set(z['sample_ids'])
                        if row['approach']=='global' and row['round']==3:
                            assert np.allclose(z['p'],learning.predict(model,name,client,client['hold_indices']),atol=1e-6)
                    checked+=1
                completed+=1
    expected=len(ELIGIBLE)*len(SEEDS)*len(DOMAINS)
    if require_complete:assert completed==expected,(completed,expected)
    assert completed>0
    result=dict(status='complete' if completed==expected else 'partial',validated_runs=completed,expected_runs=expected,
        recomputed_metric_sets=checked,global_models_and_replicas_verified=True,test_used=False,signature=signature())
    dump(FED/'validation_report.json',result);return result

def soft_domains(exp):
    if exp=='exp4':
        from .personalized import domains,transform
        return transform(domains(),final=True)
    return load_smoke()

def validate_soft(require_complete=True):
    checked=completed=0;cache={}
    for path in SOFT.glob('*/*/*/run_config.json'):
        cfg=json.loads(path.read_text())
        if cfg.get('status')!='complete' or cfg.get('signature')!=signature():continue
        if cfg['experiment'] not in cache:cache[cfg['experiment']]=soft_domains(cfg['experiment'])
        domains={ds:d for ds,d in cache[cfg['experiment']].items() if ds in cfg['models']}
        clients={c['uid']:c for d in domains.values() for c in d['clients'].values()}
        train={uid:c for uid,c in clients.items() if c['split']=='train'}
        assert not cfg['test_used'];folder=path.parent
        student=GlobalCNN1D({ds:len(d['config']['feature_names']) for ds,d in domains.items()},cfg['seed'])
        initial_hash=model_hash(student)
        student.load_state_dict(torch.load(folder/'global_final.pt',map_location='cpu',weights_only=True))
        assert model_hash(student)==cfg['global_hash']
        audits=json.loads((folder/'server_training_audit.json').read_text());assert len(audits)==3
        for i,a in enumerate(audits,1):
            assert a['round']==i and a['server_model_trained'] and not a['private_data_received'] and not a['evaluation_predictions_received']
            assert a['input_global_hash']!=(a['output_global_hash'])
            assert a['input_global_hash']==(initial_hash if i==1 else audits[i-2]['output_global_hash'])
            with np.load(folder/f'teacher_round_{i}.npz') as z:
                assert z['logits'].shape==(len(train),32,2)
                assert set(z['client_ids'])==set(train)
                actual=torch.tensor(z['logits']).softmax(-1).numpy()
                assert np.allclose(actual,z['probabilities'])
                assert np.allclose(np.average(actual,axis=0,weights=z['weights']),z['consensus'])
                for ds in domains:
                    assert np.isclose(sum(w for uid,w in zip(z['client_ids'],z['weights']) if clients[str(uid)]['dataset']==ds),1/len(domains))
                assert not set(z['anchor_ids'])&{v for c in clients.values() for v in c['sample_ids']}
            stage=GlobalCNN1D({ds:len(d['config']['feature_names']) for ds,d in domains.items()},cfg['seed'])
            stage.load_state_dict(torch.load(folder/f'global_round_{i}.pt',map_location='cpu',weights_only=True))
            assert model_hash(stage)==a['output_global_hash']
        assert cfg['global_hash']==audits[-1]['output_global_hash']
        for uid,c in train.items():
            replica=GlobalCNN1D({ds:len(d['config']['feature_names']) for ds,d in domains.items()},cfg['seed'])
            replica.load_state_dict(torch.load(folder/'distributed'/f'{uid.replace(":","_")}.pt',map_location='cpu',weights_only=True))
            assert model_hash(replica)==cfg['global_hash']
            receipts=json.loads((folder/'receipts'/f'{uid.replace(":","_")}.json').read_text());assert len(receipts)==4
            for r in range(3):
                assert receipts[r]['received_global_hash']==audits[r]['input_global_hash']
                assert receipts[r]['used_for_local_distillation']==(r>0)
            assert receipts[-1]['received_global_hash']==cfg['global_hash']
        registry=json.loads((folder/'sample_registry.json').read_text())
        assert set(registry)==set(clients)
        for uid,c in clients.items():
            assert registry[uid]['fit']==c['sample_ids'][c['fit_indices']].tolist()
            assert registry[uid]['holdout']==c['sample_ids'][c['hold_indices']].tolist()
        f=pd.read_json(folder/'metrics.json')
        for row in f.to_dict('records'):
            c=clients[row['evaluation_client']]
            with np.load(folder/row['prediction_file']) as z:
                check_metrics(row,z,c)
                if row['approach']=='global_student':
                    assert np.allclose(z['p'],predictions(student,'global_student',row['dataset'],c,c['hold_indices']),atol=1e-6)
            checked+=1
        completed+=1
    if require_complete:
        for cfg in configs():
            path=SOFT/cfg['experiment']/cfg['name']/str(cfg['seed'])/'run_config.json'
            assert path.exists() and json.loads(path.read_text()).get('status')=='complete'
        for seed in range(2026,2031):
            for rule in ['common','per_dataset','per_client']:
                p=SOFT/'exp4'/rule/str(seed)/'run_config.json';assert p.exists() and json.loads(p.read_text()).get('status')=='complete'
    assert completed>0
    result=dict(status='complete' if require_complete else 'partial',validated_pairs=completed,recomputed_metric_sets=checked,
        trained_server_student_verified=True,redistributed_checkpoints_verified=True,logits_and_consensus_verified=True,test_used=False,signature=signature())
    dump(SOFT/'validation_report.json',result);return result
