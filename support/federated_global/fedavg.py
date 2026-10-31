"""FedAvg of compatible parameters; same sampling across training seeds."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from flwr.common import parameters_to_ndarrays,ndarrays_to_parameters
from .shared import *

BaseStrategy=original_fed.ECGFedAvg
class AuditedFedAvg(BaseStrategy):
    def __init__(self,train_ids,evaluation_ids,initial,root):
        self.parameter_dtypes=[a.dtype for a in initial]
        super().__init__(train_ids,evaluation_ids,initial,root)
    def configure_fit(self,server_round,parameters,client_manager):
        arrays=parameters_to_ndarrays(parameters)
        self.current_input_hash=arrays_hash(arrays)
        if server_round>1:
            assert self.current_input_hash==self.audit[-1]['global_parameter_hash']
        return super().configure_fit(server_round,parameters,client_manager)
    def aggregate_fit(self,server_round,results,failures):
        parameters,metrics=super().aggregate_fit(server_round,results,failures)
        arrays=[a.astype(dtype) for a,dtype in zip(parameters_to_ndarrays(parameters),self.parameter_dtypes)]
        parameters=ndarrays_to_parameters(arrays)
        np.savez_compressed(self.root/f'global_round_{server_round}.npz',**{f'p_{j}':a for j,a in enumerate(arrays)})
        self.audit[-1].update(input_global_parameter_hash=self.current_input_hash,
            global_parameter_hash=arrays_hash(arrays),global_model_created=True)
        dump(self.root/'aggregation_audit.json',self.audit)
        return parameters,metrics
    def configure_evaluate(self,server_round,parameters,client_manager):
        if server_round:
            assert arrays_hash(parameters_to_ndarrays(parameters))==self.audit[-1]['global_parameter_hash']
            self.audit[-1]['redistributed_to']=self.train_ids+self.evaluation_ids
            dump(self.root/'aggregation_audit.json',self.audit)
        return super().configure_evaluate(server_round,parameters,client_manager)

def run_one(ds,name,seed):
    assert name in ELIGIBLE
    destination=FED/f'{ds}_80_64_{seed}'/name;meta=destination/'run_config.json'
    if meta.exists():
        config=json.loads(meta.read_text())
        if config.get('status')=='complete' and config.get('protocol_signature')==signature() and (destination/'distribution_audit.json').exists():
            print('CACHED FEDAVG',ds,name,seed,flush=True);return
    original_fed.DATA_ROOT=FED;original_fed.ECGFedAvg=AuditedFedAvg
    original_fed.prepare=lambda dataset,train_cap,eval_cap,training_seed,include_test:raw.prepare(dataset,train_cap,eval_cap,2026,include_test)
    args=SimpleNamespace(train_cap=80,eval_cap=64,seed=seed,rounds=3,local_epochs=1,include_test=False)
    root=original_fed.run_one(ds,name,args)
    config=json.loads(meta.read_text());config.update(protocol_id='global_model_v1',protocol_signature=signature(),
        split_seed=2026,training_seed=seed,global_definition='single aggregated model per dataset, redistributed every round',
        eligible_models=ELIGIBLE,excluded=['RandomForest','KNN','GRU5','GRU_Simple'])
    dump(meta,config)
    # Explicit saved replicas of the final global state, exactly what clients receive.
    state=torch.load(root/'global_final.pt',weights_only=True,map_location='cpu')
    _,cfg,_,_=raw.prepare(ds,80,64,2026,False)
    initial=learning.factory(name,len(cfg['feature_names']),cfg['signal_vector_samples'],seed)
    initial.load_state_dict(state);digest=model_hash(initial)
    audit=json.loads((root/'aggregation_audit.json').read_text())
    assert digest==audit[-1]['global_parameter_hash']
    for uid in config['train_clients']:save_model(root/'distributed'/f'{uid}.pt',initial)
    dump(root/'distribution_audit.json',dict(global_hash=digest,recipients=config['train_clients'],
        identical_replicas=True,rounds=3))

def run(pilot=False):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    names=['CNN1D'] if pilot else ELIGIBLE
    seeds=[2026] if pilot else SEEDS
    for seed in seeds:
        for ds in DOMAINS:
            for name in names:
                dump(FED/'status.json',dict(status='running',dataset=ds,model=name,seed=seed))
                run_one(ds,name,seed)
    report(require_complete=not pilot)

def report(require_complete=True):
    rows=[]
    for seed in SEEDS:
        for ds in DOMAINS:
            for name in ELIGIBLE:
                folder=FED/f'{ds}_80_64_{seed}'/name;p=folder/'run_config.json'
                if not p.exists():continue
                config=json.loads(p.read_text())
                if config.get('status')!='complete' or config.get('protocol_signature')!=signature():continue
                f=pd.read_json(folder/'metrics.json');f=f[(f['round']==3)&f.approach.isin(['global','local_after_fedavg_training','local_federated_external','local_only','local_only_external'])]
                for (approach,split),group in f.groupby(['approach','split']):
                    rows.append(dict(dataset=ds,model=name,seed=seed,approach=approach,split=split,
                        **{key:float(group[key].mean()) for key in ['pr_auc_ap','roc_auc','recall','far','f1']}))
    f=pd.DataFrame(rows)
    if f.empty:return
    f.to_json(FED/'per_seed.json',orient='records',double_precision=15)
    keys=['dataset','model','approach','split']
    summary=f.groupby(keys)[['pr_auc_ap','roc_auc','recall','far','f1']].mean().reset_index()
    spread=f.groupby(keys).agg(pr_auc_std=('pr_auc_ap','std'),seeds=('seed','nunique')).reset_index()
    summary.merge(spread,on=keys).to_json(FED/'summary.json',orient='records',double_precision=15)
    rank=f[(f.approach=='global')&(f.split=='train')].groupby(['dataset','model']).agg(
        pr_auc_ap=('pr_auc_ap','mean'),roc_auc=('roc_auc','mean'),seeds=('seed','nunique')).reset_index()
    expected=len(ELIGIBLE)*len(DOMAINS)*len(SEEDS)
    completed=len(f[['dataset','model','seed']].drop_duplicates())
    if require_complete:assert completed==expected and rank.seeds.eq(3).all()
    selected={ds:rank[(rank.dataset==ds)&(rank.seeds==3)].sort_values(['pr_auc_ap','roc_auc','model'],ascending=[False,False,True]).head(3).model.tolist() for ds in DOMAINS}
    dump(FED/'selection.json',dict(signature=signature(),selected=selected,
        criterion='mean AP on local holdouts of the delivered GLOBAL model; three fixed-data seeds; ROC-AUC tie-break',
        exploratory=True,excluded=['RandomForest','KNN','GRU5','GRU_Simple']))
    dump(FED/'status.json',dict(status='complete' if completed==expected else 'partial',completed=completed,expected=expected))
