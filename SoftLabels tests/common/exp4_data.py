"""Exp4: amostras ampliadas e separação por registo, antes de normalizar."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from . import data
from .data import ROOT,ARTIFACTS
BASE=ARTIFACTS/'exp4'
SPLIT_SEED=2026
CAPS={'fit':96,'selection':32,'holdout':32,'dev':128}
MIN_CLASS=5

def groups_for(root,rows,rng):
    groups=[]
    for row in rows.itertuples(index=False):
        with np.load(root/row.shard_path,allow_pickle=False) as z:
            labels=z['y'];ids=z['sample_ids'] if row.record_id=='multiple_records' else None
            partitions=[(str(ids[i]),np.array([i])) for i in range(len(labels)) if labels[i]!=3] if ids is not None else [(str(row.record_id),np.flatnonzero(labels!=3))]
            for group,ix in partitions:
                y=np.isin(labels[ix],[1,2]).astype(int)
                choices=[rng.permutation(ix[y==k])[:max(CAPS.values())].tolist() for k in [0,1]]
                groups.append(dict(group=group,path=str(row.shard_path),choices=choices,counts=np.bincount(y,minlength=2)))
    return groups


def partition(groups,rng):
    n=len(groups);assert n>=3
    counts=np.array([g['counts'] for g in groups]);best=None
    for attempt in range(256):
        order=rng.permutation(n);a=max(1,int(.6*n));b=max(a+1,int(.8*n));b=min(b,n-1)
        parts=[order[:a],order[a:b],order[b:]]
        totals=np.array([counts[p].sum(0) for p in parts])
        # Estratificação dos grupos, sem consultar qualquer previsão/métrica externa.
        score=(int((totals>0).sum()),min(totals[0,1],MIN_CLASS)+min(totals[1,1],MIN_CLASS),min(totals[2,1],8),min(totals[0,1],38)+min(totals[1,1],13))
        if best is None or score>best[0]:best=(score,parts)
    return {name:[groups[i] for i in indices] for name,indices in zip(['fit','selection','holdout'],best[1])}


def sample(groups,cap,rng):
    pool=[[(g['path'],int(i),g['group']) for g in groups for i in g['choices'][k]] for k in [0,1]]
    quota=[min(len(pool[0]),cap-round(.4*cap)),min(len(pool[1]),round(.4*cap))]
    while sum(quota)<min(cap,sum(map(len,pool))):
        k=0 if quota[0]<len(pool[0]) else 1;quota[k]+=1
    selected=[pool[k][i] for k in [0,1] for i in rng.permutation(len(pool[k]))[:quota[k]]]
    rng.shuffle(selected);return selected


def prepare():
    BASE.mkdir(parents=True,exist_ok=True);domains={};audit=[]
    for dataset,folder in [('icentia11k','icentia11k_prepared_featured'),('physionet2017','physionet2027_prepared_featured')]:
        root=ROOT/'DataPreparation/FeatureExtraction'/folder;cfg=json.loads((root/'dataset_config.json').read_text())
        manifest=pd.read_csv(root/'manifest.csv',keep_default_na=False);clients={}
        for (split,cid),rows in manifest[manifest.split.isin(['train','dev'])].groupby(['split','client_id']):
            cache=BASE/'data'/dataset/(cid+'.npz');cache.parent.mkdir(parents=True,exist_ok=True)
            if not cache.exists():
                rng=np.random.default_rng(SPLIT_SEED+int(cid.split('_')[-1]));groups=groups_for(root,rows,rng)
                parts=partition(groups,rng) if split=='train' else {'dev':groups}
                chosen=[];indices={}
                for name,gs in parts.items():
                    sel=sample(gs,CAPS[name],rng);indices[name]=np.arange(len(chosen),len(chosen)+len(sel));chosen.extend(sel)
                lookup={}
                for pos,(path,i,g) in enumerate(chosen):lookup.setdefault(path,[]).append((pos,i,g))
                out={};keys=['X','features','y','sample_ids','valid_lengths']
                for path,items in lookup.items():
                    with np.load(root/path,allow_pickle=False) as z:
                        arrays={k:z[k] for k in keys}
                        for pos,i,g in items:out[pos]={**{k:v[i] for k,v in arrays.items()},'record_group':g}
                arrays={k:np.stack([out[i][k] for i in range(len(chosen))]) for k in keys+['record_group']}
                for name in ['fit','selection','holdout','dev']:arrays[name+'_indices']=indices.get(name,np.array([],int))
                np.savez_compressed(cache,**arrays)
            with np.load(cache,allow_pickle=False) as z:c={k:z[k] for k in z.files}
            c.update(dataset=dataset,uid=dataset+':'+cid,client_id=cid,split=split,patient_id=str(rows.iloc[0].patient_id),K=2)
            c['original_y']=c['y'].copy();assert not np.any(c['original_y']==3)
            c['y']=np.isin(c['original_y'],[1,2]).astype(np.int64)
            if split=='dev':c['holdout_indices']=c['dev_indices']
            c['hold_indices']=c['holdout_indices'];c['final_fit_indices']=np.concatenate([c['fit_indices'],c['selection_indices']])
            parts=['fit','selection','holdout'] if split=='train' else ['dev']
            for i,name in enumerate(parts):
                ix=c[name+'_indices'];groups=set(c['record_group'][ix]);ids=set(c['sample_ids'][ix])
                for other in parts[i+1:]:
                    oi=c[other+'_indices'];assert not groups&set(c['record_group'][oi]);assert not ids&set(c['sample_ids'][oi])
                audit.append(dict(dataset=dataset,client=c['uid'],split=split,partition=name,n=len(ix),normal=int((c['y'][ix]==0).sum()),anomaly=int((c['y'][ix]==1).sum()),records=len(groups)))
            c['selection_eligible']=split=='train' and all(np.bincount(c['y'][c[p+'_indices']],minlength=2).min()>=MIN_CLASS for p in ['fit','selection'])
            clients[cid]=c;print('EXP4 DATA',c['uid'],split,'selection eligible',c['selection_eligible'],flush=True)
        projection=np.random.default_rng(SPLIT_SEED).normal(0,1/np.sqrt(32),(cfg['signal_vector_samples'],32)).astype(np.float32)
        blocks=[];positions={};offset=0
        for c in clients.values():
            ix=c['hold_indices'];positions[c['uid']]=np.arange(offset,offset+len(ix));offset+=len(ix);blocks.append((c,ix))
        query={k:np.concatenate([c[k][ix] for c,ix in blocks]) for k in ['X','features','y','sample_ids','valid_lengths']};query['positions']=positions
        domains[dataset]=dict(dataset=dataset,clients=clients,config=cfg,projection=projection,query=query,task='binary',K=2)
    pd.DataFrame(audit).to_json(BASE/'sample_counts.json',orient='records',indent=2)
    registry={c['uid']:dict(dataset=c['dataset'],split=c['split'],selection_eligible=c['selection_eligible'],**{p:c['sample_ids'][c[p+'_indices']].tolist() for p in ['fit','selection','holdout']},**{p+'_groups':c['record_group'][c[p+'_indices']].tolist() for p in ['fit','selection','holdout']}) for d in domains.values() for c in d['clients'].values()}
    (BASE/'sample_registry.json').write_text(json.dumps(registry,indent=2))
    return domains


def view(domain,c,stage):
    """Normalização privada fit-only; seleção nunca vê holdout/dev, nem sequer as features."""
    ix=c['fit_indices'] if stage=='selection' else c['final_fit_indices']
    mean=c['features'][ix].astype(np.float64).mean(0);scale=np.maximum(c['features'][ix].astype(np.float64).std(0),1e-6)
    transforms=dict(mean=mean,scale=scale,projection=domain['projection'])
    def transform(raw):
        out=dict(raw);out['F']=((raw['features']-mean)/scale).astype(np.float32)
        projected=np.einsum('ij,jk->ik',raw['X'].astype(np.float64),domain['projection'].astype(np.float64),optimize=False)
        out['Z']=np.column_stack([projected,out['F']]).astype(np.float32);assert np.isfinite(out['Z']).all();return out
    client=transform(c);client['fit_indices']=ix
    if stage=='selection':
        sel=c['selection_indices'];query=transform({k:c[k][sel] for k in ['X','features','y','sample_ids','valid_lengths']});query['positions']={c['uid']:np.arange(len(sel))}
    else:query=transform(domain['query'])
    saved=data.SEED;data.SEED=SPLIT_SEED
    try:anchor=data.synthetic_queries(domain['config'],transforms)
    finally:data.SEED=saved
    result=dict(domain,query=query,anchor=anchor,transforms=transforms)
    return result,client
