"""Streaming full-test evaluation of frozen global models on unseen clients."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from support.federated_global.shared import ROOT,dump,model_hash
from support.ldp_logits import training_base as core
from .experiment import BASE,DESIGN,signature,folder_for

ROOTS={'icentia11k':ROOT/'DataPreparation/FeatureExtraction/icentia11k_prepared_featured',
       'physionet2017':ROOT/'DataPreparation/FeatureExtraction/physionet2027_prepared_featured'}

def serializable(row):
    return {k:(None if isinstance(v,(float,np.floating)) and not np.isfinite(v) else v) for k,v in row.items()}

def evaluate_test(domains):
    gate=json.loads((BASE/'test_gate.json').read_text())
    assert gate['signature']==signature() and gate['training_complete'] and not gate['selection_on_test']
    out=BASE/'test';out.mkdir(parents=True,exist_ok=True)
    report=out/'validation.json'
    if report.exists():
        d=json.loads(report.read_text())
        if d.get('status')=='complete' and d.get('signature')==signature():return
    models={}
    for seed in DESIGN['seeds']:
        model=core.soft.GlobalCNN1D({ds:len(d['config']['feature_names']) for ds,d in domains.items()},seed)
        model.load_state_dict(torch.load(folder_for(seed)/'global_round_3.pt',weights_only=True,map_location='cpu'))
        meta=json.loads((folder_for(seed)/'run.json').read_text())
        assert model_hash(model)==meta['final_global_hash'];model.eval();models[seed]=model
    rows=[];registry=[];pooled={seed:[] for seed in models}
    for ds,root in ROOTS.items():
        manifest=pd.read_csv(root/'manifest.csv',keep_default_na=False)
        test=manifest[manifest.split=='test'].sort_values(['client_id','shard_path'])
        assert len(test)>0
        # Test client IDs were not used for training or dev.
        assert not set(test.client_id)&set(manifest[manifest.split.isin(['train','dev'])].client_id)
        fit=np.concatenate([c['features'][c['fit_indices']] for c in domains[ds]['clients'].values() if c['split']=='train'])
        mean=fit.astype(float).mean(0);scale=np.maximum(fit.astype(float).std(0),1e-6)
        fit_ids=set(str(i) for c in domains[ds]['clients'].values() for i in c['sample_ids'])
        dataset_buffers={seed:[] for seed in models}
        for cid,shards in test.groupby('client_id',sort=True):
            ys=[];ids=[];probabilities={seed:[] for seed in models};excluded=0
            for item in shards.itertuples(index=False):
                path=root/item.shard_path
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                assert digest==item.sha256,'Prepared test shard changed'
                with np.load(path,allow_pickle=False) as z:
                    labels=z['y'];keep=np.flatnonzero(labels!=3);excluded+=int((labels==3).sum())
                    signals=z['X'];features=z['features'];lengths=z['valid_lengths'];all_ids=z['sample_ids']
                    assert set(np.unique(labels))<={0,1,2,3}
                    for start in range(0,len(keep),64):
                        ix=keep[start:start+64]
                        sample_ids=np.asarray(all_ids[ix]).astype(str)
                        assert not set(sample_ids)&fit_ids,'Test sample appeared in train/dev cache'
                        c=dict(X=signals[ix].astype(np.float32),
                               F=((features[ix]-mean)/scale).astype(np.float32),
                               valid_lengths=lengths[ix])
                        y=np.isin(labels[ix],[1,2]).astype(np.int64)
                        ys.append(y);ids.append(sample_ids)
                        for seed,model in models.items():
                            p=core.soft.predictions(model,'global_student',ds,c,np.arange(len(ix)))
                            assert np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-6)
                            probabilities[seed].append(p)
                registry.append(dict(dataset=ds,client=cid,shard=item.shard_path,sha256=digest,n_eligible=len(keep)))
                dump(out/'status.json',dict(status='evaluating',dataset=ds,client=cid,shards_completed=len(registry)))
                print('TEST SHARD COMPLETE',ds,cid,item.shard_path,flush=True)
            y=np.concatenate(ys);sample_ids=np.concatenate(ids)
            assert len(set(sample_ids))==len(sample_ids)
            for seed in models:
                p=np.concatenate(probabilities[seed]);metrics,cm=core.evaluate(y,p,ds)
                dest=out/'predictions'/str(seed);dest.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest/(ds+'_'+cid+'.npz'),p=p,y=y,sample_ids=sample_ids,cm=cm)
                rows.append(serializable(dict(seed=seed,dataset=ds,client=cid,scope='client',role='global',split='test',excluded_noise=excluded,**metrics)))
                dataset_buffers[seed].append((y,p));pooled[seed].append((y,p))
            # Persist progress, but never select checkpoints from these partial results.
            dump(out/'metrics.json',rows)
            from .notebook import refresh
            refresh()
        for seed,buffers in dataset_buffers.items():
            y=np.concatenate([v[0] for v in buffers]);p=np.concatenate([v[1] for v in buffers])
            metrics,_=core.evaluate(y,p,ds)
            rows.append(serializable(dict(seed=seed,dataset=ds,client='all',scope='dataset_pooled',role='global',split='test',**metrics)))
    for seed,buffers in pooled.items():
        y=np.concatenate([v[0] for v in buffers]);p=np.concatenate([v[1] for v in buffers]);metrics,_=core.evaluate(y,p,'combined')
        rows.append(serializable(dict(seed=seed,dataset='combined',client='all',scope='all_pooled',role='global',split='test',**metrics)))
    dump(out/'metrics.json',rows);dump(out/'data_registry.json',registry)
    # Independently reproduce all client metrics from saved predictions.
    for row in rows:
        if row['scope']!='client':continue
        path=out/'predictions'/str(row['seed'])/(row['dataset']+'_'+row['client']+'.npz')
        with np.load(path,allow_pickle=False) as z:
            metrics,_=core.evaluate(z['y'],z['p'],row['dataset'])
        for metric in core.METRICS:
            actual=metrics[metric];stored=row[metric]
            assert (stored is None and not np.isfinite(actual)) or np.isclose(actual,stored,atol=1e-12)
    dump(report,dict(status='complete',signature=signature(),seed_count=len(models),
        evaluated='all prepared test examples except original noise class 3',normalization='training fit only',
        fixed_round=3,fixed_threshold=.5,selection_on_test=False,local_models_on_unseen_clients=False,
        n_examples_once=sum(r['n_eligible'] for r in registry),test_metrics_recomputed=True))
