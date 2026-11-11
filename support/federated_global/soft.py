"""Teachers -> anchor logits -> trained server CNN1D -> identical student replicas."""
import copy,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import flwr as fl
from flwr.common import FitIns,ndarrays_to_parameters,parameters_to_ndarrays
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg
from .shared import *
from global_fed.metrics import evaluate

class GlobalCNN1D(nn.Module):
    """One shared CNN trunk/classifier; source adapters handle 17 vs 19 features."""
    def __init__(self,feature_dims,seed):
        super().__init__();torch.manual_seed(seed)
        self.trunk=learning.models.CNNTrunk()
        self.tabular=nn.ModuleDict({ds:nn.Sequential(nn.Linear(n,16),nn.ReLU()) for ds,n in feature_dims.items()})
        self.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(80,2))
    def forward(self,ds,x,features,lengths):
        z=self.trunk(x)
        mask=learning.models.temporal_mask(lengths,z.shape[-1],input_steps=x.shape[-1])[:,None,:]
        signal=(z*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        return self.classifier(torch.cat([signal,self.tabular[ds](features)],dim=1))

def forward(model,name,ds,c,ix):
    if name=='global_student':
        return model(ds,torch.from_numpy(c['X'][ix,None,:]),torch.from_numpy(c['F'][ix]),torch.from_numpy(c['valid_lengths'][ix]))
    learning.models.SIGNAL_LENGTH=c['X'].shape[1]
    return learning.logits(model,name,c,ix)

def predictions(model,name,ds,c,ix,as_logits=False):
    model.eval();values=[]
    with torch.no_grad():
        for begin in range(0,len(ix),32):
            out=forward(model,name,ds,c,ix[begin:begin+32]);values.append((out if as_logits else out.softmax(1)).numpy())
    return np.concatenate(values)

def soft_update(model,name,ds,anchor,target,seed,epochs=1):
    torch.manual_seed(seed);model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    target=torch.tensor(target,dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out=forward(model,name,ds,anchor,np.arange(len(target)))
        loss=-(target*F.log_softmax(out,dim=1)).sum(1).mean()
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()

def teacher_initial(ds,c,name,seed,protocol):
    path=SOFT/'warmup'/protocol[:16]/str(seed)/ds/name/(c['client_id']+'.pt')
    m=learning.factory(name,c['F'].shape[1],c['X'].shape[1],seed)
    if path.exists():m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));return m
    # Initial teachers have no supervised epochs; each round adds one private epoch.
    save_model(path,m);return m

class TeacherClient(fl.client.NumPyClient):
    def __init__(self,ds,c,name,seed,anchor,student,protocol):
        self.ds,self.c,self.name,self.seed,self.anchor=ds,c,name,seed,anchor
        self.teacher=teacher_initial(ds,c,name,seed,protocol)
        self.student=copy.deepcopy(student);self.receipts=[]
    def fit(self,parameters,config):
        r=int(config['round']);learning.set_parameters(self.student,parameters)
        digest=model_hash(self.student);assert digest==str(config['global_hash'])
        if r>1:
            target=predictions(self.student,'global_student',self.ds,self.anchor,np.arange(32))
            soft_update(self.teacher,self.name,self.ds,self.anchor,target,self.seed+r)
        learning.models.SIGNAL_LENGTH=self.c['X'].shape[1]
        learning.fit(self.teacher,self.name,self.c,1,self.seed+r,
                     ae_epochs=2 if self.name=='Autoencoder_CNN_GradCAM' and r==1 else 0)
        self.receipts.append(dict(round=r,received_global_hash=digest,used_for_local_distillation=r>1))
        logits=predictions(self.teacher,self.name,self.ds,self.anchor,np.arange(32),as_logits=True)
        return [logits],len(self.c['fit_indices']),{}

class ServerDistillation(FedAvg):
    def __init__(self,clients,domains,student,folder,seed):
        self.clients,self.domains,self.student,self.folder,self.seed=clients,domains,student,folder,seed
        self.audit=[];self.last_hash=model_hash(student)
        super().__init__(initial_parameters=ndarrays_to_parameters(learning.get_parameters(student)),
            min_fit_clients=len(clients),min_available_clients=len(clients),accept_failures=False)
    def configure_fit(self,server_round,parameters,client_manager):
        assert arrays_hash(parameters_to_ndarrays(parameters))==self.last_hash
        return [(client_manager.all()[uid],FitIns(parameters,dict(round=server_round,global_hash=self.last_hash))) for uid in sorted(self.clients)]
    def aggregate_fit(self,server_round,results,failures):
        assert not failures and {p.cid for p,_ in results}==set(self.clients)
        ids=[];logits=[];probabilities=[];weights=[]
        for proxy,result in sorted(results,key=lambda x:x[0].cid):
            arrays=parameters_to_ndarrays(result.parameters);assert len(arrays)==1 and arrays[0].shape==(32,2)
            z=arrays[0];assert np.isfinite(z).all()
            p=torch.from_numpy(z).softmax(1).numpy();ds=self.clients[proxy.cid].ds
            n=sum(c.ds==ds for c in self.clients.values())
            ids.append(proxy.cid);logits.append(z);probabilities.append(p);weights.append(1/(len(self.domains)*n))
        consensus=np.average(probabilities,weights=weights,axis=0).astype(np.float32)
        before=model_hash(self.student)
        # Server receives only logits. No private signal, label or evaluation prediction is used here.
        torch.manual_seed(self.seed+100+server_round)
        self.student.train();optimizer=torch.optim.AdamW(self.student.parameters(),lr=.001,weight_decay=.0001)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss=sum(-(torch.tensor(consensus)*F.log_softmax(forward(self.student,'global_student',ds,d['anchor'],np.arange(32)),dim=1)).sum(1).mean() for ds,d in self.domains.items())/len(self.domains)
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(self.student.parameters(),5);optimizer.step()
        after=model_hash(self.student);assert before!=after
        save_model(self.folder/f'global_round_{server_round}.pt',self.student)
        np.savez_compressed(self.folder/f'teacher_round_{server_round}.npz',logits=np.stack(logits),
            probabilities=np.stack(probabilities),consensus=consensus,weights=weights,client_ids=ids,
            anchor_ids=next(iter(self.domains.values()))['anchor']['sample_ids'])
        self.audit.append(dict(round=server_round,input_global_hash=before,output_global_hash=after,
            server_model_trained=True,student_architecture='shared CNN1D; source feature adapters',
            payload='anchor logits only',temperature=1,private_data_received=False,
            evaluation_predictions_received=False,client_ids=ids,weights=weights))
        dump(self.folder/'server_training_audit.json',self.audit);self.last_hash=after
        return ndarrays_to_parameters(learning.get_parameters(self.student)),{}
    def configure_evaluate(self,server_round,parameters,client_manager):return []
    def evaluate(self,server_round,parameters):return None

def record(rows,folder,model,name,ds,c,approach,owner):
    ix=c['hold_indices'];p=predictions(model,name,ds,c,ix);y=c['y'][ix]
    metrics,cm=evaluate(y,p,ds);filename=f'prediction_{len(rows):05}.npz'
    np.savez_compressed(folder/filename,y=y,p=p,cm=cm,sample_ids=c['sample_ids'][ix])
    rows.append(dict(dataset=ds,client=owner,evaluation_client=c['uid'],split=c['split'],approach=approach,
        prediction_file=filename,**metrics))

def run_pair(cfg,domains):
    seed=cfg['seed'];token=signature();folder=SOFT/cfg['experiment']/cfg['name']/str(seed)
    folder.mkdir(parents=True,exist_ok=True);meta=folder/'run_config.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        if old.get('status')=='complete' and old.get('signature')==token:
            print('CACHED SOFT',cfg['experiment'],cfg['name'],seed,flush=True);return
    start=time.monotonic();dump(meta,dict(cfg,status='running',signature=token,test_used=False,rounds=3,
        global_definition='trained CNN1D server student; not probability ensemble',
        local_definition='heterogeneous teacher; received student used next round',
        student='CNN1D shared trunk/head and source feature adapters',server_epochs_per_round=3,
        redistribution='full student state_dict to every client each round and final'))
    dump(folder/'sample_registry.json',sample_registry(domains))
    student=GlobalCNN1D({ds:len(d['config']['feature_names']) for ds,d in domains.items()},seed)
    train={c['uid']:(ds,c) for ds,d in domains.items() for c in d['clients'].values() if c['split']=='train'}
    clients={};controls={};manager=SimpleClientManager()
    for uid,(ds,c) in train.items():
        name=cfg['assignments'].get(uid,cfg['models'].get(ds))
        assert name in ELIGIBLE
        instance=TeacherClient(ds,c,name,seed,domains[ds]['anchor'],student,token)
        clients[uid]=instance;controls[uid]=copy.deepcopy(instance.teacher)
        manager.register(original_fed.MemoryProxy(uid,instance))
    strategy=ServerDistillation(clients,domains,student,folder,seed)
    server=fl.server.Server(client_manager=manager,strategy=strategy);server.set_max_workers(1)
    server.fit(num_rounds=3,timeout=None)
    learning.set_parameters(student,parameters_to_ndarrays(server.parameters))
    global_hash=model_hash(student);assert global_hash==strategy.last_hash
    rows=[]
    for uid,client in clients.items():
        ds,c=train[uid]
        learning.set_parameters(client.student,learning.get_parameters(student))
        assert model_hash(client.student)==global_hash
        save_model(folder/'distributed'/f'{uid.replace(":","_")}.pt',client.student)
        client.receipts.append(dict(round='final',received_global_hash=global_hash,used_for_local_distillation=False))
        dump(folder/'receipts'/f'{uid.replace(":","_")}.json',client.receipts)
        control=controls[uid]
        for r in range(1,4):
            # Same private epochs and corresponding extra update steps, without public targets.
            if r>1:learning.fit(control,client.name,c,1,seed+r)
            learning.fit(control,client.name,c,1,seed+r,ae_epochs=2 if client.name=='Autoencoder_CNN_GradCAM' and r==1 else 0)
        save_model(folder/'teachers'/f'{uid.replace(":","_")}.pt',client.teacher)
        for target in domains[ds]['clients'].values():
            if target['split']=='train' and target['uid']!=uid:continue
            for approach,model in [('local_teacher',client.teacher),('local_control',control)]:
                record(rows,folder,model,client.name,ds,target,approach,uid)
    for ds,d in domains.items():
        for c in d['clients'].values():record(rows,folder,student,'global_student',ds,c,'global_student','server')
    save_model(folder/'global_final.pt',student)
    pd.DataFrame(rows).to_json(folder/'metrics.json',orient='records',double_precision=15)
    metadata=json.loads(meta.read_text());metadata.update(status='complete',global_hash=global_hash,
        final_recipients=sorted(clients),elapsed_seconds=time.monotonic()-start,
        control_budget='three supervised epochs plus two extra private epochs vs two public-KD steps; FLOPs/examples/time not matched')
    dump(meta,metadata);print('COMPLETE SOFT',cfg['experiment'],cfg['name'],seed,round(metadata['elapsed_seconds'],1),flush=True)

def shortlist():
    selected=json.loads((FED/'selection.json').read_text())
    assert selected['signature']==signature() and all(len(selected['selected'][ds])==3 for ds in DOMAINS)
    return selected['selected']

def configs():
    chosen=shortlist();union=list(dict.fromkeys(chosen['icentia11k']+chosen['physionet2017']))
    result=[]
    def add(exp,name,models,seeds):
        for s in seeds:result.append(dict(experiment=exp,name=name,models=models,assignments={},seed=s))
    for model in union:
        for ds in DOMAINS:add('exp1',ds+'__'+model,{ds:model},[2026])
    cross=list(itertools.product(chosen['icentia11k'],chosen['physionet2017']))
    for a,b in cross:add('exp2',f'IC_{a}__PH_{b}',dict(zip(DOMAINS,[a,b])),[2026])
    for a,b in itertools.product(union,repeat=2):add('exp3',f'IC_{a}__PH_{b}',dict(zip(DOMAINS,[a,b])),SEEDS)
    pairs=cross+[(m,m) for m in union if (m,m) not in cross]
    for a,b in pairs:add('exp5',f'IC_{a}__PH_{b}',dict(zip(DOMAINS,[a,b])),SEEDS)
    return result

def run(experiments=None,pilot=False):
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    domains=load_smoke()
    if pilot:
        configurations=[dict(experiment='pilot',name='IC_CNN1D__PH_CNN_BiLSTM',models=dict(zip(DOMAINS,['CNN1D','CNN_BiLSTM'])),assignments={},seed=2026)]
    else:configurations=[c for c in configs() if experiments is None or c['experiment'] in experiments]
    for cfg in configurations:
        dump(SOFT/'status.json',dict(status='running',experiment=cfg['experiment'],configuration=cfg['name'],seed=cfg['seed']))
        run_pair(cfg,{ds:domains[ds] for ds in cfg['models']})
    if not pilot and (experiments is None or 'exp4' in experiments):
        from .personalized import run as run_personalized
        run_personalized(domains)
    dump(SOFT/'status.json',dict(status='pilot_complete' if pilot else 'training_complete',signature=signature()))
