"""Flower Server + NumPyClient + FedAvg, transporte em memória, execução sequencial."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/icentia-mpl')
os.environ.setdefault('FLWR_TELEMETRY_ENABLED','0')
from pathlib import Path
import argparse,copy,json,time,hashlib
import numpy as np
import pandas as pd
import torch
import flwr as fl
from flwr.common import FitIns,EvaluateIns,DisconnectRes,ndarrays_to_parameters,parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from .data import prepare,ROOT,DATA_ROOT
from .learning import MODELS,LOCAL_ONLY,factory,get_parameters,set_parameters,fit,predict
from .metrics import evaluate

class MemoryProxy(ClientProxy):
    """Transporte de simulação: mantém as mensagens/API de clientes Flower reais."""
    def __init__(self,cid,client):super().__init__(cid);self.client=client.to_client()
    def get_properties(self,ins,timeout,group_id):return self.client.get_properties(ins)
    def get_parameters(self,ins,timeout,group_id):return self.client.get_parameters(ins)
    def fit(self,ins,timeout,group_id):return self.client.fit(ins)
    def evaluate(self,ins,timeout,group_id):return self.client.evaluate(ins)
    def reconnect(self,ins,timeout,group_id):return DisconnectRes(reason='simulation complete')

class Recorder:
    def __init__(self,root,dataset,model):self.root=root;self.dataset=dataset;self.model=model;self.rows=[];self.counter=0
    def record(self,m,c,kind,round_id,owner=None):
        ix=c['hold_indices'];p=predict(m,self.model,c,ix);y=c['y'][ix]
        metrics,cm=evaluate(y,p,self.dataset);self.counter+=1
        token=f'{self.counter:05d}'
        np.savez_compressed(self.root/f'prediction_{token}.npz',y=y,p=p,sample_ids=c['sample_ids'][ix],cm=cm)
        row=dict(dataset=self.dataset,model=self.model,approach=kind,round=round_id,
            model_client=owner or 'server',evaluation_client=c['client_id'],split=c['split'],
            patient_id=c['patient_id'],prediction_file=f'prediction_{token}.npz',**metrics)
        self.rows.append(row)
        pd.DataFrame(self.rows).to_json(self.root/'metrics.json',orient='records',double_precision=15)
        return metrics,p

class ECGClient(fl.client.NumPyClient):
    def __init__(self,c,name,initial,rec,args):
        self.c=c;self.name=name;self.m=copy.deepcopy(initial);self.rec=rec;self.args=args;self.last_local=None
    def get_parameters(self,config):return get_parameters(self.m)
    def fit(self,parameters,config):
        assert self.c['split']=='train','Dev/test nunca podem treinar.'
        set_parameters(self.m,parameters);round_id=int(config['round'])
        fit(self.m,self.name,self.c,self.args.local_epochs,self.args.seed+round_id,
            ae_epochs=2 if round_id==1 else 0)
        self.last_local=get_parameters(self.m)
        self.rec.record(self.m,self.c,'local_after_fedavg_training',round_id,self.c['client_id'])
        return self.last_local,len(self.c['fit_indices']),{}
    def evaluate(self,parameters,config):
        assert self.c['split']!='test' or config.get('final_test',False)
        set_parameters(self.m,parameters)
        metrics,p=self.rec.record(self.m,self.c,'global',int(config['round']))
        ix=self.c['hold_indices'];loss=float(-np.log(np.clip(p[np.arange(len(ix)),self.c['y'][ix]],1e-12,1)).mean())
        return loss,len(ix),{k:float(v) for k,v in metrics.items() if np.isfinite(v)}

def weighted_client_metrics(results):
    # Resumo do histórico Flower; AUC pooled é calculada separadamente das previsões.
    keys=set().union(*(metrics.keys() for _,metrics in results)) if results else set()
    values={}
    for key in keys:
        valid=[(n,m[key]) for n,m in results if key in m and np.isfinite(m[key])]
        if valid:values['client_weighted_mean_'+key]=sum(n*v for n,v in valid)/sum(n for n,_ in valid)
    return values

class ECGFedAvg(FedAvg):
    def __init__(self,train_ids,evaluation_ids,initial,root):
        super().__init__(fraction_fit=1,fraction_evaluate=1,min_fit_clients=len(train_ids),
            min_evaluate_clients=len(evaluation_ids),min_available_clients=len(train_ids)+len(evaluation_ids),
            initial_parameters=ndarrays_to_parameters(initial),accept_failures=False,inplace=False,
            fit_metrics_aggregation_fn=lambda results: {},evaluate_metrics_aggregation_fn=weighted_client_metrics)
        self.train_ids=train_ids;self.evaluation_ids=evaluation_ids;self.root=root;self.audit=[]
    def configure_fit(self,server_round,parameters,client_manager):
        clients=client_manager.all();ins=FitIns(parameters,{'round':server_round})
        return [(clients[cid],ins) for cid in self.train_ids]
    def configure_evaluate(self,server_round,parameters,client_manager):
        clients=client_manager.all();ins=EvaluateIns(parameters,{'round':server_round})
        return [(clients[cid],ins) for cid in self.train_ids+self.evaluation_ids]
    def aggregate_fit(self,server_round,results,failures):
        assert not failures and len(results)==len(self.train_ids)
        assert {p.cid for p,_ in results}==set(self.train_ids)
        aggregated,metrics=super().aggregate_fit(server_round,results,failures)
        arrays=parameters_to_ndarrays(aggregated);total=sum(r.num_examples for _,r in results)
        decoded=[(r.num_examples,parameters_to_ndarrays(r.parameters)) for _,r in results]
        for j,actual in enumerate(arrays):
            expected=sum(values[j]*weight for weight,values in decoded)/total
            assert np.allclose(actual,expected,rtol=1e-5,atol=1e-6)
        self.audit.append(dict(round=server_round,clients={p.cid:r.num_examples for p,r in results},
                               n_examples=total,weighted_average_verified=True))
        (self.root/'aggregation_audit.json').write_text(json.dumps(self.audit,indent=2))
        return aggregated,metrics

def gradcam(model,c,root):
    ix=c['hold_indices'][:1];cache={}
    def capture(module,inputs,out):cache['a']=out
    handle=model.cam_target().register_forward_hook(capture)
    try:
        model.eval();x=torch.from_numpy(c['X'][ix,None,:]);f=torch.from_numpy(c['F'][ix]);length=torch.from_numpy(c['valid_lengths'][ix])
        out=model(x,f,length);target=int(out.argmax(1)[0]);a=cache['a']
        g=torch.autograd.grad(out[0,target],a)[0]
        cam=(g.mean(-1,keepdim=True)*a).sum(1,keepdim=True).relu()
        cam=torch.nn.functional.interpolate(cam,size=x.shape[-1],mode='linear',align_corners=False)
        cam=cam/cam.amax(-1,keepdim=True).clamp_min(1e-8)
        np.savez_compressed(root/'global_gradcam_dev.npz',heatmap=cam.detach().numpy(),sample_ids=c['sample_ids'][ix],explained_class=target)
    finally:handle.remove()

def run_one(dataset,name,args):
    root=DATA_ROOT/f'{dataset}_{args.train_cap}_{args.eval_cap}_{args.seed}'/name
    root.mkdir(parents=True,exist_ok=True)
    clients,cfg,registry,transforms=prepare(dataset,args.train_cap,args.eval_cap,args.seed,args.include_test)
    registry.to_json(root/'client_registry.json',orient='records')
    np.savez_compressed(root/'preprocessing.npz',**transforms)
    train_ids=sorted(k for k,c in clients.items() if c['split']=='train')
    dev_ids=sorted(k for k,c in clients.items() if c['split']=='dev')
    test_ids=sorted(k for k,c in clients.items() if c['split']=='test')
    initial=factory(name,len(cfg['feature_names']),cfg['signal_vector_samples'],args.seed)
    rec=Recorder(root,dataset,name);start=time.time()
    metadata=dict(task='binary',protocol_id='binary_smoke_v3',positive_class=1,dataset=dataset,model=name,flwr_version=fl.__version__,rounds=args.rounds,local_epochs=args.local_epochs,
        learning_rate=0.0003 if name=='CNN1D_Optimized' else 0.001,weight_decay=0.0001,batch_size=32,autoencoder_extra_epochs=2,
        train_cap=args.train_cap,eval_cap=args.eval_cap,seed=args.seed,train_clients=train_ids,dev_clients=dev_ids,test_clients=test_ids,
        transport='sequential in-process Flower ClientProxy; real Flower Server.fit and FedAvg; no Ray/network deployment',
        test_used=args.include_test,aggregation='none: local reference' if name in LOCAL_ONLY else 'Flower FedAvg weighted by local training examples',
        sampling='binary_smoke_v3: fixed shared_smoke_v2 IDs; exclude noise/Q;  per-client class quotas, up to 4 examples per available class, proportional remainder; 20% internal holdout; prevalence enriched',
        anomaly_definition='A/O vs N; noise excluded' if dataset=='physionet2017' else 'S/V vs N; Q excluded',
        classical_representation='fixed shared Gaussian projection 32 + train-only standardized features; not centralized PCA',
        svm='linear multiclass hinge; softmax margins are uncalibrated scores' if name=='SVM' else None,
        gru_temporal_pool_hop=max(1,(cfg['signal_vector_samples']+499)//500) if name in {'GRU5','GRU_Simple'} else None,
        cnn_optimized_protocol='fixed lr=0.0003; no FL hyperparameter search' if name=='CNN1D_Optimized' else None,
        checkpoint_selection='fixed final round, no tuning on test',status='running')
    (root/'run_config.json').write_text(json.dumps(metadata,indent=2))
    # Baseline independente por cliente, com o mesmo orçamento supervisionado.
    for cid in train_ids:
        baseline=copy.deepcopy(initial);c=clients[cid]
        fit(baseline,name,c,args.rounds*args.local_epochs,args.seed+1,ae_epochs=2)
        rec.record(baseline,c,'local_only',args.rounds,cid)
        for eid in dev_ids+test_ids:rec.record(baseline,clients[eid],'local_only_external',args.rounds,cid)
    if name not in LOCAL_ONLY:
        manager=SimpleClientManager();instances={}
        for cid,c in clients.items():
            instances[cid]=ECGClient(c,name,initial,rec,args);manager.register(MemoryProxy(cid,instances[cid]))
        strategy=ECGFedAvg(train_ids,dev_ids,get_parameters(initial),root)
        server=fl.server.Server(client_manager=manager,strategy=strategy);server.set_max_workers(1)
        history,elapsed=server.fit(num_rounds=args.rounds,timeout=None)
        global_parameters=parameters_to_ndarrays(server.parameters)
        global_model=copy.deepcopy(initial);set_parameters(global_model,global_parameters)
        torch.save(global_model.state_dict(),root/'global_final.pt')
        # Test só é avaliado depois da última ronda fixa, sem seleção ou ajuste.
        for eid in test_ids:instances[eid].evaluate(global_parameters,{'round':args.rounds,'final_test':True})
        for cid in train_ids:
            local=copy.deepcopy(initial);set_parameters(local,instances[cid].last_local)
            torch.save(local.state_dict(),root/f'{cid}_local_final.pt')
            for eid in dev_ids+test_ids:rec.record(local,clients[eid],'local_federated_external',args.rounds,cid)
        if name=='Autoencoder_CNN_GradCAM':gradcam(global_model,clients[dev_ids[0]],root)
        assert len(strategy.audit)==args.rounds
    else:
        # KNN/RF: treino local coordenado pelo Flower e agregação de previsões.
        from .ensemble import run_ensemble
        run_ensemble(dataset,name,args,weighting='equal')
    # Métricas globais pooled: recalculadas a partir de todas as previsões, não média de AUCs.
    records=pd.DataFrame(rec.rows)
    for split in ['train','dev','test']:
        group=records[(records.approach=='global')&(records['round']==args.rounds)&(records.split==split)]
        if group.empty:continue
        ys=[];ps=[]
        for row in group.itertuples(index=False):
            with np.load(root/row.prediction_file,allow_pickle=False) as z:ys.append(z['y']);ps.append(z['p'])
        values,cm=evaluate(np.concatenate(ys),np.concatenate(ps),dataset)
        rec.rows.append(dict(dataset=dataset,model=name,approach='global_pooled',round=args.rounds,model_client='server',evaluation_client='ALL',split=split,**values))
    pd.DataFrame(rec.rows).to_json(root/'metrics.json',orient='records',double_precision=15)
    metadata.update(status='complete',elapsed_seconds=time.time()-start)
    (root/'run_config.json').write_text(json.dumps(metadata,indent=2))
    print('COMPLETE',dataset,name,round(metadata['elapsed_seconds'],1),flush=True)
    return root

def main(default_model=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['physionet2017','icentia11k','both'],default='both')
    parser.add_argument('--model',choices=MODELS+['all'],default=default_model or 'all')
    parser.add_argument('--rounds',type=int,default=3);parser.add_argument('--local-epochs',type=int,default=1)
    parser.add_argument('--train-cap',type=int,default=80);parser.add_argument('--eval-cap',type=int,default=64)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args();args.include_test=False
    assert args.rounds>0 and args.local_epochs>0 and args.train_cap>=20 and args.eval_cap>=4
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    datasets=['physionet2017','icentia11k'] if args.dataset=='both' else [args.dataset]
    names=MODELS if args.model=='all' else [args.model]
    for dataset in datasets:
        for name in names:run_one(dataset,name,args)
    from .report import build_report
    build_report(args)
    from .ensemble import summarize_ensembles
    summarize_ensembles(args)
