"""Seleção shared_smoke_v2 e âncoras sintéticas sem exemplos reais de avaliação."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
# Não importar 'common' dos testes FedAvg: o pacote desta experiência tem o mesmo nome.
import importlib.util

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

legacy_data=load_module('soft_legacy_data',ROOT/'FedAvg tests/common/data.py')
models=load_module('soft_native_models',ROOT/'FedAvg tests/common/models.py')
features_module=load_module('soft_signal_features',ROOT/'DataPreparation/FeatureExtraction/build_featured_datasets.py')
SEED=2026
MODELS=['GRU5', 'CNN_BiLSTM', 'GRU_Simple', 'RandomForest', 'KNN']
NAMES={'MobileNetV2':'MobileNetV2 1D','RandomForest':'Random Forest','CNN1D':'CNN 1D','KNN':'KNN','GRU5':'GRU de cinco camadas','GRU_Simple':'GRU simplificada','CNN_BiLSTM':'CNN 1D + BiLSTM'}
ARTIFACTS=ROOT/'SoftLabels tests' / 'artifacts'

def synthetic_queries(config,transforms,n=32):
    """Mesmo ID/latente por âncora, duas vistas sintéticas nas grelhas nativas."""
    rng=np.random.default_rng(SEED+917)
    length=config['signal_vector_samples'];is_beat=config['source_dataset']=='icentia11k'
    rows=[];raw_rows=[]
    for i in range(n):
        rng=np.random.default_rng(SEED+917+i*31)
        width=rng.uniform(.012,.065);amp=rng.uniform(.6,2.5);noise=rng.uniform(.005,.15)
        skew=rng.uniform(-.08,.08);ripple=rng.uniform(0,.18);rr=rng.uniform(.5,1.5)
        # Vistas aproximadas de batimento e registo: não são ECG clínicos validados.
        t=np.linspace(-.5,.5,length) if is_beat else np.arange(length)/200
        phase=t if is_beat else ((t+rr/2)%rr)-rr/2
        x=amp*(np.exp(-((phase-skew)/width)**2)-.3*np.exp(-((phase-skew-.07)/(.8*width))**2))
        x+=ripple*np.sin(2*np.pi*7*t)+noise*rng.normal(size=length)
        raw=x.copy();x=(x-x.mean())/max(x.std(),1e-8)
        grid=features_module.beat_feature_grid(x[None,:])[0] if is_beat else x
        out=features_module.scalar_features(grid,include_rr=not is_beat)
        out['source_peak_to_peak']=float(np.ptp(raw));out['source_max_abs']=float(np.abs(raw).max())
        raw_rows.append([out.get(name,np.nan) for name in config['feature_names']]);rows.append(x.astype(np.float32))
    X=np.stack(rows);raw=np.array(raw_rows)
    raw=np.where(np.isfinite(raw),raw,transforms['mean'])
    F=np.clip((raw-transforms['mean'])/transforms['scale'],-8,8).astype(np.float32)
    projected=np.einsum('ij,jk->ik',X.astype(np.float64),transforms['projection'].astype(np.float64),optimize=False)
    Z=np.column_stack([projected,F]).astype(np.float32)
    assert np.isfinite(Z).all()
    return dict(X=X,F=F,Z=Z,valid_lengths=np.full(n,length,dtype=np.int64),sample_ids=np.array([f'synthetic_anchor_{i:03}' for i in range(n)]))

def load_domain(dataset,task):
    assert task=='binary', 'Todas as experiências ativas usam normal vs anomalia.'
    clients,cfg,registry,transforms=legacy_data.prepare(dataset,80,64,SEED,False,binary=False)
    for c in clients.values():
        c['dataset']=dataset;c['uid']=dataset+':'+c['client_id']
        c['original_y']=c['y'].copy()
        if task=='binary':
            c['fit_indices']=c['fit_indices'][c['y'][c['fit_indices']]!=3]
            c['hold_indices']=c['hold_indices'][c['y'][c['hold_indices']]!=3]
            c['y']=np.isin(c['y'],[1,2]).astype(np.int64)
        c['K']=2 if task=='binary' else 4
    if task=='binary':
        # Reajustar a normalização apenas sobre fit elegível, excluindo ~ e Q também destas estatísticas.
        features=np.concatenate([c['features'][c['fit_indices']] for c in clients.values() if c['split']=='train'])
        features=features.astype(np.float64)
        transforms['mean']=features.mean(0);transforms['scale']=np.maximum(features.std(0),1e-6)
        for c in clients.values():
            c['F']=((c['features']-transforms['mean'])/transforms['scale']).astype(np.float32)
            projection=np.einsum('ij,jk->ik',c['X'].astype(np.float64),transforms['projection'].astype(np.float64),optimize=False)
            c['Z']=np.column_stack([projection,c['F']]).astype(np.float32)
    anchor=synthetic_queries(cfg,transforms)
    # Consultas de avaliação por domínio; os rótulos ficam fora da comunicação fit.
    blocks=[];positions={};offset=0
    for uid,c in clients.items():
        ix=c['hold_indices'];positions[c['uid']]=np.arange(offset,offset+len(ix));offset+=len(ix)
        blocks.append((c,ix))
    query={key:np.concatenate([c[key][ix] for c,ix in blocks]) for key in ['X','F','Z','valid_lengths','sample_ids','y']}
    query['positions']=positions
    assert not set(query['sample_ids'])&set(anchor['sample_ids'])
    fit_ids=set().union(*(set(c['sample_ids'][c['fit_indices']]) for c in clients.values() if c['split']=='train'))
    assert not fit_ids&set(query['sample_ids'])
    return dict(dataset=dataset,clients=clients,config=cfg,anchor=anchor,query=query,task=task,K=2 if task=='binary' else 4)


def signature(domain,c,model):
    payload=dict(protocol='softlabels_v1',task=domain['task'],dataset=c['dataset'],model=model,uid=c['uid'],
        fit_ids=c['sample_ids'][c['fit_indices']].tolist(),feature_names=domain['config']['feature_names'],warmup_epochs=3,
        seed=SEED,query_ids=domain['query']['sample_ids'].tolist(),anchor_hash=hashlib.sha256(domain['anchor']['X'].tobytes()).hexdigest(),
        data_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        models_hash=hashlib.sha256((ROOT/'FedAvg tests/common/models.py').read_bytes()).hexdigest(),
        adapter_hash=hashlib.sha256((Path(__file__).parent/'local.py').read_bytes()).hexdigest())
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
