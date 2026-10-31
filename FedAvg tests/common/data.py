"""Amostras determinísticas por cliente; test só é aberto quando solicitado."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parents[2]
RAW_DATA_ROOT=ROOT/'FedAvg tests' / 'artifacts'/'shared_smoke_v2'
DATA_ROOT=ROOT/'FedAvg tests' / 'artifacts'/'binary_smoke_v3'

def prepare(dataset, train_cap=80, eval_cap=64, seed=2026, include_test=False,binary=True):
    folder='physionet2027_prepared_featured' if dataset=='physionet2017' else 'icentia11k_prepared_featured'
    root=ROOT/'DataPreparation'/'FeatureExtraction'/folder
    cfg=json.loads((root/'dataset_config.json').read_text())
    shards=pd.read_csv(root/'manifest.csv',keep_default_na=False)
    expected={'physionet2017':{'train':6,'dev':1,'test':1},'icentia11k':{'train':16,'dev':2,'test':2}}[dataset]
    assert shards.groupby('split').client_id.nunique().to_dict()==expected
    if dataset=='icentia11k':
        assert shards.groupby('client_id').patient_id.nunique().eq(1).all()
        sets=[set(shards.loc[shards.split==s,'patient_id']) for s in ['train','dev','test']]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    if include_test:
        raise ValueError('Este protocolo usa exclusivamente train/dev; test não pode ser aberto.')
    cache=RAW_DATA_ROOT/'cache'/f'{dataset}_{train_cap}_{eval_cap}_{seed}_0'
    cache.mkdir(parents=True,exist_ok=True)
    selected_shards=shards[shards.split.isin(['train','dev'])]
    registry=selected_shards.groupby(['split','client_id','patient_id'],dropna=False).n_examples.sum().reset_index()
    registry.to_json(cache/'registry.json',orient='records')
    clients={}
    for (split,cid),group in selected_shards.groupby(['split','client_id']):
        dest=cache/f'{cid}.npz'
        if not dest.exists():
            rng=np.random.default_rng(seed+int(cid.split('_')[-1]))
            cap=train_cap if split=='train' else eval_cap
            # Reservoir por classe: conserva apenas metadados dos candidatos, não todos os sinais.
            reservoirs={k:[] for k in range(4)}
            for row in group.itertuples(index=False):
                with np.load(root/row.shard_path,allow_pickle=False) as z:
                    y=z['y']
                    for k in range(4):
                        ix=np.flatnonzero(y==k)
                        if not len(ix):continue
                        keys=rng.random(len(ix));take=min(cap,len(ix))
                        chosen=np.argpartition(keys,take-1)[:take]
                        reservoirs[k].extend((float(keys[j]),str(row.shard_path),int(ix[j])) for j in chosen)
                        reservoirs[k]=sorted(reservoirs[k])[:cap]
            # Até quatro exemplos por classe disponível; restante orçamento proporcional à fonte.
            count_table=pd.read_csv(root/'client_class_counts.csv')
            class_names=list(cfg['label_to_id'])
            available=count_table.loc[count_table.client_id==cid,class_names].sum().to_numpy(dtype=int)
            budget=min(cap,int(available.sum()));quota=np.minimum(available,4)
            assert quota.sum()<=budget
            while quota.sum()<budget:
                weights=np.where(quota<available,available,0).astype(float)
                shares=(budget-quota.sum())*weights/weights.sum()
                add=np.minimum(np.floor(shares).astype(int),available-quota)
                if not add.sum():add[np.argmax(np.where(quota<available,shares,-1))]=1
                quota+=add
            chosen=[item for k in range(4) for item in reservoirs[k][:quota[k]]]
            rng.shuffle(chosen)
            arrays={k:[] for k in ['X','features','y','sample_ids','valid_lengths']}
            # Reagrupar por shard evita descomprimir repetidamente o mesmo ficheiro.
            lookup={}
            for position,(_,path,index) in enumerate(chosen):lookup.setdefault(path,[]).append((position,index))
            collected={}
            for path,items in lookup.items():
                with np.load(root/path,allow_pickle=False) as z:
                    for pos,index in items:collected[pos]={key:z[key][index] for key in arrays}
            arrays={key:np.stack([collected[pos][key] for pos in range(len(chosen))]) for key in arrays}
            ix=np.arange(len(chosen))
            if split=='train':
                # Holdout exato de 20%; classes únicas ficam em fit e classes com >=2 têm holdout.
                y=arrays['y'];counts=np.bincount(y,minlength=4)
                hold_quota=np.where(counts>=2,1,0);hold_budget=round(.2*len(ix))
                while hold_quota.sum()<hold_budget:
                    eligible=hold_quota<np.maximum(counts-1,0)
                    target=.2*counts-hold_quota
                    hold_quota[np.argmax(np.where(eligible,target,-np.inf))]+=1
                hold=np.concatenate([rng.permutation(np.flatnonzero(y==k))[:hold_quota[k]] for k in range(4)])
                fit=np.setdiff1d(ix,hold);hold=np.sort(hold)
            else:fit=np.array([],dtype=int);hold=ix
            np.savez_compressed(dest,**arrays,fit_indices=fit,hold_indices=hold)
        with np.load(dest,allow_pickle=False) as z:client={k:z[k] for k in z.files}
        client.update(split=split,client_id=cid,patient_id=str(group.iloc[0].patient_id))
        assert not set(client['fit_indices']) & set(client['hold_indices'])
        clients[cid]=client
    ids=[set(c['sample_ids']) for c in clients.values()]
    assert sum(map(len,ids))==len(set().union(*ids))
    if binary:
        cfg=dict(cfg,original_label_to_id=cfg['label_to_id'],label_to_id={'normal':0,'anomaly':1},task='binary',protocol_id='binary_smoke_v3')
        for c in clients.values():
            c['original_y']=c['y'].copy()
            c['fit_indices']=c['fit_indices'][c['y'][c['fit_indices']]!=3]
            c['hold_indices']=c['hold_indices'][c['y'][c['hold_indices']]!=3]
            c['y']=np.isin(c['y'],[1,2]).astype(np.int64)
    # Estatísticas globais calculadas exclusivamente a partir dos subconjuntos locais de treino.
    train=[c for c in clients.values() if c['split']=='train']
    count=sum(len(c['fit_indices']) for c in train)
    sums=sum(np.nan_to_num(c['features'][c['fit_indices']].astype(float)).sum(0) for c in train)
    squares=sum(np.square(np.nan_to_num(c['features'][c['fit_indices']].astype(float))).sum(0) for c in train)
    assert all(np.isfinite(c['features']).all() for c in clients.values()), 'Suporte a missing features precisa de extensão das estatísticas.'
    mean=sums/count;scale=np.sqrt(np.maximum(squares/count-mean**2,1e-12))
    if binary:
        values=np.concatenate([c['features'][c['fit_indices']] for c in train]).astype(np.float64)
        mean=values.mean(0);scale=np.maximum(values.std(0),1e-6)
    # Projeção pública fixa: dispensa PCA centralizada e preserva alinhamento entre clientes.
    rng=np.random.default_rng(seed)
    projection=rng.normal(0,1/np.sqrt(32),(cfg['signal_vector_samples'],32)).astype(np.float32)
    for c in clients.values():
        c['F']=((c['features']-mean)/scale).astype(np.float32)
        if binary:p=np.einsum('ij,jk->ik',c['X'].astype(np.float64),projection.astype(np.float64),optimize=False)
        else:
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):p=c['X']@projection
        c['Z']=np.column_stack([p,c['F']]).astype(np.float32)
        assert np.isfinite(c['Z']).all()
    return clients,cfg,registry,{'mean':mean,'scale':scale,'projection':projection}
