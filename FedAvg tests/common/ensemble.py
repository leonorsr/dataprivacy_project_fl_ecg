"""One-shot federated ensembles: Flower clients train locally; Strategy averages predictions."""
import argparse,copy,json,time
import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import FitIns,EvaluateIns,ndarrays_to_parameters,parameters_to_ndarrays
from flwr.server.strategy import Strategy
from flwr.server.client_manager import SimpleClientManager
from .data import prepare,DATA_ROOT
from .learning import factory,fit,predict,LOCAL_ONLY
from .metrics import evaluate
from .experiment import MemoryProxy

class VotingClient(fl.client.NumPyClient):
    def __init__(self,c,name,initial,positions,root,seed):
        self.c=c;self.name=name;self.model=copy.deepcopy(initial)
        self.positions=positions;self.root=root;self.seed=seed;self.local_predictions=None
    def get_parameters(self,config):return []
    def fit(self,parameters,config):
        assert self.c['split']=='train'
        # parameters contém só features de consulta, sem labels ou exemplos de fit alheios.
        assert len(parameters)==1
        fit(self.model,self.name,self.c,1,self.seed)
        query=parameters[0];p=np.zeros((len(query),2),dtype=float)
        p[:,self.model.classes_.astype(int)]=self.model.predict_proba(query)
        assert np.allclose(p.sum(1),1)
        self.local_predictions=p.copy()
        # Checkpoint local: não é enviado ao servidor, nem se enviam vizinhos/árvores.
        import joblib
        joblib.dump(self.model,self.root/f'{self.c["client_id"]}_local.joblib')
        return [p],len(self.c['fit_indices']),{}
    def evaluate(self,parameters,config):
        assert self.c['split']!='test' or config.get('final_test',False)
        p=parameters[0][self.positions];y=self.c['y'][self.c['hold_indices']]
        values,_=evaluate(y,p,config['dataset'])
        loss=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean())
        return loss,len(y),{k:float(v) for k,v in values.items() if np.isfinite(v)}

class SoftVoting(Strategy):
    """Agrega probabilidades alinhadas por exemplo/classe; não calcula médias de modelos."""
    def __init__(self,query,train_ids,dev_ids,dataset,weighting,root):
        self.query=query;self.train_ids=train_ids;self.dev_ids=dev_ids
        self.dataset=dataset;self.weighting=weighting;self.root=root;self.audit=None
    def initialize_parameters(self,client_manager):return ndarrays_to_parameters([self.query])
    def configure_fit(self,server_round,parameters,client_manager):
        assert server_round==1,'Ensemble one-shot: não repetir rondas sem novos dados/modelos.'
        return [(client_manager.all()[cid],FitIns(parameters,{})) for cid in self.train_ids]
    def aggregate_fit(self,server_round,results,failures):
        assert not failures and {p.cid for p,_ in results}==set(self.train_ids)
        results=sorted(results,key=lambda pair:pair[0].cid)
        probabilities=[parameters_to_ndarrays(r.parameters)[0] for _,r in results]
        assert all(p.shape==(len(self.query),2) and np.isfinite(p).all() and np.allclose(p.sum(1),1) for p in probabilities)
        weights=np.array([1 if self.weighting=='equal' else r.num_examples for _,r in results],float)
        weights/=weights.sum()
        aggregated=np.tensordot(weights,np.stack(probabilities),axes=1)
        assert np.allclose(aggregated,sum(w*p for w,p in zip(weights,probabilities)))
        self.audit=dict(strategy='soft voting of predictions, not FedAvg',rounds=1,
            training_clients=[p.cid for p,_ in results],fit_examples=[r.num_examples for _,r in results],
            weighting=self.weighting,weights=weights.tolist(),n_queries=len(self.query),
            class_order_aligned=True,probability_average_verified=True)
        (self.root/'aggregation_audit.json').write_text(json.dumps(self.audit,indent=2))
        return ndarrays_to_parameters([aggregated]),{}
    def configure_evaluate(self,server_round,parameters,client_manager):
        return [(client_manager.all()[cid],EvaluateIns(parameters,{'dataset':self.dataset})) for cid in self.train_ids+self.dev_ids]
    def aggregate_evaluate(self,server_round,results,failures):
        assert not failures
        return sum(r.loss*r.num_examples for _,r in results)/sum(r.num_examples for _,r in results),{}
    def evaluate(self,server_round,parameters):return None

def run_ensemble(dataset,name,args,weighting='equal'):
    assert name in LOCAL_ONLY and weighting in {'equal','examples'}
    root=DATA_ROOT/f'{dataset}_{args.train_cap}_{args.eval_cap}_{args.seed}'/name/f'ensemble_{weighting}'
    root.mkdir(parents=True,exist_ok=True)
    clients,cfg,registry,transforms=prepare(dataset,args.train_cap,args.eval_cap,args.seed,args.include_test)
    registry.to_json(root/'client_registry.json',orient='records')
    np.savez_compressed(root/'preprocessing.npz',**transforms)
    train_ids=sorted(cid for cid,c in clients.items() if c['split']=='train')
    dev_ids=sorted(cid for cid,c in clients.items() if c['split']=='dev')
    test_ids=sorted(cid for cid,c in clients.items() if c['split']=='test')
    # Consultas idênticas e ordenadas para todos os modelos locais; nunca contêm fit.
    query=[];positions={};ids=[];offset=0
    for cid,c in clients.items():
        ix=c['hold_indices'];positions[cid]=np.arange(offset,offset+len(ix));offset+=len(ix)
        query.append(c['Z'][ix]);ids.extend(c['sample_ids'][ix].tolist())
    assert len(ids)==len(set(ids))
    fit_ids=set().union(*(set(c['sample_ids'][c['fit_indices']]) for c in clients.values()))
    assert not fit_ids&set(ids)
    query=np.concatenate(query);initial=factory(name,len(cfg['feature_names']),cfg['signal_vector_samples'],args.seed)
    manager=SimpleClientManager();instances={};started=time.time()
    config=dict(dataset=dataset,model=name,weighting=weighting,rounds=1,flwr_version=fl.__version__,
        train_clients=train_ids,dev_clients=dev_ids,test_clients=test_ids,test_used=args.include_test,
        train_cap=args.train_cap,eval_cap=args.eval_cap,seed=args.seed,
        aggregation='Flower custom Strategy, federated soft-voting ensemble',
        protocol='one local fit, then probability aggregation; local models unchanged by aggregation',
        query_protocol='shared heldout feature matrix in this in-process simulation; never query labels; not a private inference protocol',
        class_names=list(cfg['label_to_id']),status='running')
    (root/'run_config.json').write_text(json.dumps(config,indent=2))
    for cid,c in clients.items():
        instances[cid]=VotingClient(c,name,initial,positions[cid],root,args.seed+1)
        manager.register(MemoryProxy(cid,instances[cid]))
    strategy=SoftVoting(query,train_ids,dev_ids,dataset,weighting,root)
    server=fl.server.Server(client_manager=manager,strategy=strategy);server.set_max_workers(1)
    server.fit(num_rounds=1,timeout=None)
    assert strategy.audit is not None
    p_global=parameters_to_ndarrays(server.parameters)[0]
    for cid in test_ids:instances[cid].evaluate([p_global],{'final_test':True,'dataset':dataset})
    rows=[];counter=0
    def record(p,y,sample_ids,approach,split,cid,owner='ensemble'):
        nonlocal counter
        values,cm=evaluate(y,p,dataset);counter+=1;file=f'prediction_{counter:04d}.npz'
        np.savez_compressed(root/file,p=p,y=y,sample_ids=np.array(sample_ids),cm=cm)
        rows.append(dict(dataset=dataset,model=name,weighting=weighting,approach=approach,split=split,
            evaluation_client=cid,model_client=owner,prediction_file=file,**values))
    for cid,c in clients.items():
        ix=c['hold_indices'];pos=positions[cid]
        record(p_global[pos],c['y'][ix],c['sample_ids'][ix],'ensemble',c['split'],cid)
        if cid in train_ids:
            p=instances[cid].local_predictions
            record(p[pos],c['y'][ix],c['sample_ids'][ix],'local_only',c['split'],cid,cid)
            for eid in dev_ids+test_ids:
                ec=clients[eid];eix=ec['hold_indices'];epos=positions[eid]
                record(p[epos],ec['y'][eix],ec['sample_ids'][eix],'local_only_external',ec['split'],eid,cid)
    for split in ['train','dev','test']:
        cids=[cid for cid,c in clients.items() if c['split']==split]
        if not cids:continue
        pos=np.concatenate([positions[cid] for cid in cids])
        y=np.concatenate([clients[cid]['y'][clients[cid]['hold_indices']] for cid in cids])
        record(p_global[pos],y,np.array(ids)[pos],'ensemble_pooled',split,'ALL')
    np.savez_compressed(root/'ensemble_predictions.npz',p=p_global,sample_ids=np.array(ids))
    # Permite verificar a agregação e os ganhos locais, sem voltar a treinar.
    np.savez_compressed(root/'client_query_predictions.npz',
        probabilities=np.stack([instances[cid].local_predictions for cid in train_ids]),
        client_ids=np.array(train_ids),sample_ids=np.array(ids),weights=np.array(strategy.audit['weights']))
    pd.DataFrame(rows).to_json(root/'metrics.json',orient='records',double_precision=15)
    config.update(status='complete',elapsed_seconds=time.time()-started)
    (root/'run_config.json').write_text(json.dumps(config,indent=2))
    print('COMPLETE ENSEMBLE',dataset,name,weighting,flush=True)
    return root

def summarize_ensembles(args):
    frames=[]
    for dataset in ['physionet2017','icentia11k']:
        parent=DATA_ROOT/f'{dataset}_{args.train_cap}_{args.eval_cap}_{args.seed}'
        for path in sorted(parent.glob('*/ensemble_*/run_config.json')):
            config=json.loads((path.parent/'run_config.json').read_text())
            if config['status']=='complete':frames.append(pd.read_json(path.parent/'metrics.json') if (path.parent/'metrics.json').exists() else pd.read_csv(path.parent/'metrics.csv'))
    if not frames:return
    metrics=pd.concat(frames,ignore_index=True)
    metrics.to_json(DATA_ROOT/'ensemble_results.json',orient='records',double_precision=15)
    local=metrics[(metrics.split=='train')&metrics.approach.isin(['local_only','ensemble'])]
    fields=['pr_auc_ap','roc_auc','f1','precision','recall','miss_rate','far','pr_auc_ap_macro_ovr','roc_auc_macro_ovr','accuracy','f1_macro','precision_macro','recall_macro','miss_rate_macro','far_macro','anomaly_pr_auc_ap','anomaly_roc_auc','anomaly_recall','anomaly_miss_rate','anomaly_far','auc_valid_classes']
    summary=local.groupby(['dataset','model','weighting','approach'])[fields].mean().reset_index()
    summary.to_json(DATA_ROOT/'ensemble_local_comparison.json',orient='records',double_precision=15)
    from .data import ROOT
    path=ROOT/'FedAvg tests'/'analise_comparativa.txt'
    marker='\n\n6. KNN E RANDOM FOREST — AGREGAÇÃO FEDERADA DE PREVISÕES\n'
    existing=path.read_text().split(marker)[0] if path.exists() else ''
    text=marker+'Flower Strategy personalizada: cada cliente treina o modelo local; média das probabilidades alinhadas por exemplo e classe. Não é FedAvg.\nUma ronda one-shot; repetir o mesmo fit não cria aprendizagem colaborativa adicional. A agregação não modifica os modelos locais.\nPesos equal: mesmo peso por cliente; examples: proporcional ao número de exemplos fit. Após excluir ~ / Q, o número de fit pode diferir por cliente; pesos iguais e por exemplos podem produzir resultados distintos.\nConsulta partilhada de features de avaliação nesta simulação; labels das consultas não são enviadas aos modelos. Não é inferência privada.\n\nMÉTRICAS LOCAIS: média com igual peso por cliente no mesmo holdout.\n'+summary.round(4).to_string(index=False)
    pooled=metrics[(metrics.approach=='ensemble_pooled')&metrics.split.isin(['dev','test'])]
    text+='\n\nGENERALIZAÇÃO DO ENSEMBLE EM DEV/TEST (POOLED)\n'+pooled[['dataset','model','weighting','split']+fields].round(4).to_string(index=False)+'\n'
    path.write_text(existing+text)

def main(default_model=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['both','physionet2017','icentia11k'],default='both')
    parser.add_argument('--model',choices=['both','KNN','RandomForest'],default=default_model or 'both')
    parser.add_argument('--weighting',choices=['both','equal','examples'],default='both')
    parser.add_argument('--train-cap',type=int,default=80);parser.add_argument('--eval-cap',type=int,default=64)
    parser.add_argument('--seed',type=int,default=2026);parser.set_defaults(include_test=False)
    args=parser.parse_args()
    for dataset in (['physionet2017','icentia11k'] if args.dataset=='both' else [args.dataset]):
        for model in (['KNN','RandomForest'] if args.model=='both' else [args.model]):
            for weighting in (['equal','examples'] if args.weighting=='both' else [args.weighting]):run_ensemble(dataset,model,args,weighting)
    summarize_ensembles(args)
