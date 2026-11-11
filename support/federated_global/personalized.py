"""Grouped exp4: selection per client, then a trained server student for every rule."""
import copy,importlib.util,json,shutil,sys
import numpy as np
import pandas as pd
from .shared import *
from .soft import predictions,run_pair,shortlist

def domains():
    spec=importlib.util.spec_from_file_location('global_soft.global_grouped_data',ROOT/'SoftLabels tests/common/exp4_data.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.BASE=SOFT/'exp4_data';source=ROOT/'SoftLabels tests/artifacts/exp4/data'
    assert source.exists()
    for path in source.glob('*/*.npz'):
        dest=module.BASE/'data'/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copy2(path,dest)
    return module.prepare()

def transform(raw_domains,final=False):
    out={}
    for ds,d in raw_domains.items():
        feature_rows=np.concatenate([c['features'][c['final_fit_indices'] if final else c['fit_indices']] for c in d['clients'].values() if c['split']=='train'])
        mean=feature_rows.astype(float).mean(0);scale=np.maximum(feature_rows.astype(float).std(0),1e-6)
        projection=d['projection'];clients={}
        for uid,c in d['clients'].items():
            a=dict(c);a['F']=((c['features']-mean)/scale).astype(np.float32)
            projected=np.einsum('ij,jk->ik',c['X'].astype(float),projection.astype(float),optimize=False)
            a['Z']=np.column_stack([projected,a['F']]).astype(np.float32)
            if c['split']=='train' and final:a['fit_indices']=c['final_fit_indices']
            clients[uid]=a
        anchor=old_soft.synthetic_queries(d['config'],dict(mean=mean,scale=scale,projection=projection))
        out[ds]=dict(d,clients=clients,anchor=anchor)
    return out

def select(raw_domains):
    path=SOFT/'exp4_selection.json'
    if path.exists():
        selected=json.loads(path.read_text())
        if selected.get('signature')==signature():return selected
    chosen=shortlist();union=list(dict.fromkeys(chosen[DOMAINS[0]]+chosen[DOMAINS[1]]))
    data=transform(raw_domains,final=False);rows=[]
    for ds,d in data.items():
        for c in d['clients'].values():
            if c['split']!='train' or not c['selection_eligible']:continue
            for name in union:
                file=SOFT/'selection_predictions'/ds/c['client_id']/(name+'.npz');file.parent.mkdir(parents=True,exist_ok=True)
                model=learning.factory(name,c['F'].shape[1],c['X'].shape[1],2026)
                learning.fit(model,name,c,3,2027,ae_epochs=2 if name=='Autoencoder_CNN_GradCAM' else 0)
                p=predictions(model,name,ds,c,c['selection_indices']);y=c['y'][c['selection_indices']]
                from global_fed.metrics import evaluate
                m,_=evaluate(y,p,ds)
                np.savez_compressed(file,y=y,p=p,sample_ids=c['sample_ids'][c['selection_indices']])
                rows.append(dict(dataset=ds,client=c['uid'],model=name,pr_auc_ap=m['pr_auc_ap'],roc_auc=m['roc_auc']))
                print('GLOBAL EXP4 CANDIDATE',c['uid'],name,flush=True)
    f=pd.DataFrame(rows);f.to_json(SOFT/'exp4_selection_metrics.json',orient='records',double_precision=15)
    def choose(frame):return frame.sort_values(['pr_auc_ap','roc_auc','model'],ascending=[False,False,True]).iloc[0].model
    scores=f.groupby(['dataset','model'])[['pr_auc_ap','roc_auc']].mean().reset_index()
    common=choose(scores.groupby('model')[['pr_auc_ap','roc_auc']].mean().reset_index())
    per={ds:choose(scores[scores.dataset==ds]) for ds in DOMAINS};assignments={r:{} for r in ['common','per_dataset','per_client']}
    for ds,d in data.items():
        for c in d['clients'].values():
            if c['split']!='train':continue
            uid=c['uid'];assignments['common'][uid]=common;assignments['per_dataset'][uid]=per[ds]
            assignments['per_client'][uid]=choose(f[f.client==uid]) if c['selection_eligible'] else per[ds]
    selected=dict(signature=signature(),models=union,common=common,per_dataset=per,assignments=assignments,
        selection_seed=2026,selection_fixed=True,holdout_used=False,dev_used=False,test_used=False,
        normalization='source fit-only shared coordinates; final refit uses fit+selection, never holdout/dev')
    dump(path,selected);return selected

def run(unused_smoke=None):
    data=domains();selected=select(data);final=transform(data,final=True)
    for seed in range(2026,2031):
        for rule in ['common','per_dataset','per_client']:
            cfg=dict(experiment='exp4',name=rule,models={ds:selected['per_dataset'][ds] for ds in DOMAINS},assignments=selected['assignments'][rule],seed=seed)
            dump(SOFT/'status.json',dict(status='running',experiment='exp4',configuration=rule,seed=seed))
            run_pair(cfg,final)
