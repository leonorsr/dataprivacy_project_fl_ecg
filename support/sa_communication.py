"""Paired end-to-end FedAvg/soft-label experiments, with and without secure sums."""
import copy,fcntl,hashlib,importlib.util,json,os,shutil,sys,time,traceback
from pathlib import Path
os.environ.setdefault('FLWR_TELEMETRY_ENABLED','0')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/sa-communication-mpl')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from support.federated_global import shared,soft,personalized
from support.federated_global.shared import ROOT,dump,learning,model_hash,save_model,sample_registry
from global_fed.metrics import evaluate

BASE=ROOT/'Privacy Techniques/SA/Communication'
CACHE=ROOT/'artifacts/sa_communication'
SPEC=importlib.util.spec_from_file_location('_communication_sa',BASE.parent/'protocol.py')
SA=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(SA)
TEACHERS={'icentia11k':'Autoencoder_CNN_GradCAM','physionet2017':'MobileNetV2'}
METRICS=['pr_auc_ap','roc_auc','recall','far','f1']
DESIGN=dict(version=1,seeds=list(range(2031,2036)),rounds=3,local_epochs=10,server_steps=3,
    bits=24,threshold=12,fedavg_public_bound=256.0,teachers=TEACHERS,global_model='shared CNN1D with source feature adapters',
    weighting='fit examples / total fit examples',local_metric='fit-weighted own-client holdout AP before aggregation',
    global_metric='fit-weighted mean AP on the same client holdouts; not pooled AP',
    primary_round=3,dropouts=False,test_used=False,split_seed=2026,
    communication='protocol payload bytes, not network traffic; fresh SA setup per round',
    baseline_encoding='float32 values including numeric integer buffers; public fit count 4 bytes',
    broadcast='initial state and one new global state after each round; final broadcast counted once',
    anchors='32 synthetic anchors generated locally from shared recipe; no raw private data transmitted',
    limitations='paired SA ablations isolate SA; FedAvg vs distillation also changes local architecture and objective; no matched FLOPs')

def freeze():
    BASE.mkdir(parents=True,exist_ok=True)
    path=BASE/'design.json'
    if path.exists():assert json.loads(path.read_text())==DESIGN,'Design changed: use a new experiment version'
    else:dump(path,DESIGN)

def signature():
    files=[Path(__file__),BASE.parent/'protocol.py',ROOT/'support/federated_global/soft.py',
           ROOT/'support/federated_global/personalized.py',ROOT/'SoftLabels tests/common/exp4_data.py',
           ROOT/'SoftLabels tests/common/data.py']+[ROOT/'FedAvg tests/common'/n for n in ['models.py','learning.py','metrics.py']]
    return hashlib.sha256(json.dumps(DESIGN,sort_keys=True).encode()+b''.join(p.read_bytes() for p in files)).hexdigest()

def prepare():
    """Copy the already fixed grouped sample, never resample or modify previous experiments."""
    source=ROOT/'SoftLabels tests/final candidates/artifacts/data/data'
    module_path=ROOT/'SoftLabels tests/common/exp4_data.py'
    spec=importlib.util.spec_from_file_location('global_soft._communication_data',module_path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.BASE=CACHE/'data'
    for p in source.glob('*/*.npz'):
        target=module.BASE/'data'/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(p,target)
        assert hashlib.sha256(target.read_bytes()).digest()==hashlib.sha256(p.read_bytes()).digest()
    domains=personalized.transform(module.prepare(),final=True)
    registry=sample_registry(domains)
    train=[c for d in domains.values() for c in d['clients'].values() if c['split']=='train']
    assert len(train)==22 and all(len(c['fit_indices'])==128 and len(c['hold_indices'])==32 for c in train)
    for c in train:assert not set(c['sample_ids'][c['fit_indices']])&set(c['sample_ids'][c['hold_indices']])
    reference=json.loads((ROOT/'SoftLabels tests/final candidates/artifacts/sample_registry.json').read_text())
    assert registry==reference,'Client sample IDs changed'
    dump(BASE/'sample_registry.json',registry)
    dump(BASE/'data_manifest.json',dict(source=str(source.relative_to(ROOT)),test_used=False,
        fit_examples={ds:sum(len(c['fit_indices']) for c in d['clients'].values() if c['split']=='train') for ds,d in domains.items()},
        registry_sha256=hashlib.sha256(json.dumps(registry,sort_keys=True).encode()).hexdigest()))
    return domains

def supervised_global(model,ds,c,epochs,seed):
    torch.manual_seed(seed);rng=np.random.default_rng(seed);ix=c['fit_indices'];y=c['y'][ix]
    counts=np.bincount(y,minlength=2);w=np.zeros(2,dtype=np.float32);present=counts>0
    w[present]=len(y)/(present.sum()*counts[present])
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    for _ in range(epochs):
        order=rng.permutation(ix)
        for start in range(0,len(order),32):
            batch=order[start:start+32]
            if len(batch)<2:continue
            opt.zero_grad(set_to_none=True)
            logits=soft.forward(model,'global_student',ds,c,batch)
            loss=F.cross_entropy(logits,torch.from_numpy(c['y'][batch]).long(),weight=torch.from_numpy(w))
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()

def aggregate(messages,counts,use_sa,kind,bits=24,threshold=12,bound=64.):
    """Encode signed FedAvg states with a PUBLIC affine bound, soft targets in [0,1]."""
    x=np.asarray(messages,dtype=np.float64);counts=np.asarray(counts,dtype=float);w=counts/counts.sum()
    assert x.ndim==2 and np.isfinite(x).all() and (counts>0).all()
    reference=np.average(x,axis=0,weights=w)
    start=time.perf_counter();plain_up=4*x.size+4*len(x)
    if use_sa:
        if kind=='Fedavg':
            assert np.max(np.abs(x))<=bound,'FedAvg state exceeds public bound; do not silently clip'
            encoded=(x+bound)/(2*bound)
        else:
            assert x.min()>=0 and x.max()<=1
            encoded=x
        q={i+1:SA.quantize(w[i]*encoded[i],bits) for i in range(len(x))}
        total,report=SA.secure_sum(q,threshold)
        assert report['survivors']==len(x)
        # Verify the CRYPTOGRAPHIC result independently of fixed-point error.
        assert np.array_equal(total,sum(q.values())%SA.MOD)
        result=SA.dequantize(total,bits)
        if kind=='Fedavg':result=2*bound*result-bound
        upload=4*len(x)+int(round(report['upload_bytes_mean']*len(x)))
        download=int(round(report['download_bytes_mean']*len(x)))
        # Includes key generation/encoding, unlike the internal protocol wall timer.
        server_seconds=report['server_seconds']
    else:result=np.average(x,axis=0,weights=w);upload=plain_up;download=0;server_seconds=0.
    return result,dict(upload_bytes=upload,sa_download_bytes=download,baseline_upload_bytes=plain_up,
        aggregation_seconds=time.perf_counter()-start,server_unmask_seconds=server_seconds,
        aggregate_mae=float(np.abs(result-reference).mean()),aggregate_max_error=float(np.abs(result-reference).max()),
        secure_integer_sum_verified=bool(use_sa),values_per_client=x.shape[1])

def pack(model):return np.concatenate([a.ravel().astype(np.float64) for a in learning.get_parameters(model)])

def unpack(model,vector):
    arrays=[];offset=0
    for a in learning.get_parameters(model):
        out=vector[offset:offset+a.size].reshape(a.shape)
        if np.issubdtype(a.dtype,np.integer):out=np.rint(out)
        arrays.append(out.astype(a.dtype));offset+=a.size
    assert offset==len(vector);learning.set_parameters(model,arrays)

def record(rows,predictions_dir,model,name,ds,c,role,round_):
    p=soft.predictions(model,name,ds,c,c['hold_indices']);y=c['y'][c['hold_indices']]
    metrics,cm=evaluate(y,p,ds);file=f'{len(rows):05}.npz'
    predictions_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(predictions_dir/file,p=p,y=y,cm=cm,sample_ids=c['sample_ids'][c['hold_indices']])
    rows.append(dict(round=round_,dataset=ds,client=c['uid'],split='holdout' if c['split']=='train' else 'dev',
        role=role,fit_examples=len(c['fit_indices']),prediction_file=file,**metrics))

def run_one(domains,kind,use_sa,seed,settings=None,base=None):
    cfg=dict(DESIGN if settings is None else settings);base=BASE if base is None else Path(base)
    condition='SA' if use_sa else 'plain';folder=base/kind/'artifacts'/str(seed)/condition
    folder.mkdir(parents=True,exist_ok=True);token=signature();meta=folder/'run.json'
    if meta.exists():
        previous=json.loads(meta.read_text())
        if previous.get('status')=='complete' and previous['signature']==token and previous['settings']==cfg:
            validate_run(folder,domains);return
    (folder/'validation.json').unlink(missing_ok=True)
    start=time.perf_counter();dump(meta,dict(status='running',signature=token,settings=cfg,kind=kind,condition=condition,seed=seed,test_used=False))
    train={c['uid']:(ds,c) for ds,d in domains.items() for c in d['clients'].values() if c['split']=='train'}
    ids=sorted(train);counts=np.array([len(train[uid][1]['fit_indices']) for uid in ids]);weights=counts/counts.sum()
    source_weights={ds:sum(weights[i] for i,uid in enumerate(ids) if train[uid][0]==ds) for ds in domains}
    global_model=soft.GlobalCNN1D({ds:len(d['config']['feature_names']) for ds,d in domains.items()},seed)
    teachers={}
    if kind=='Softlabels':
        for uid in ids:
            ds,c=train[uid];teachers[uid]=learning.factory(TEACHERS[ds],c['F'].shape[1],c['X'].shape[1],seed)
    metrics=[];communication=[];audit=[]
    for r in range(1,cfg['rounds']+1):
        dump(base/'status.json',dict(status='running',kind=kind,condition=condition,seed=seed,round=r,signature=token))
        print('TRAIN',kind,condition,'seed',seed,'round',r,flush=True)
        before=model_hash(global_model);local={};messages=[];training_seconds=0.
        for index,uid in enumerate(ids):
            ds,c=train[uid];begin=time.perf_counter()
            if kind=='Fedavg':
                model=copy.deepcopy(global_model);assert model_hash(model)==before
                supervised_global(model,ds,c,cfg['local_epochs'],seed+r);messages.append(pack(model));name='global_student'
            else:
                model=teachers[uid];name=TEACHERS[ds]
                if r>1:
                    target=soft.predictions(global_model,'global_student',ds,domains[ds]['anchor'],np.arange(32))
                    soft.soft_update(model,name,ds,domains[ds]['anchor'],target,seed+r)
                learning.fit(model,name,c,cfg['local_epochs'],seed+r,ae_epochs=2 if name=='Autoencoder_CNN_GradCAM' and r==1 else 0)
                probs=soft.predictions(model,name,ds,domains[ds]['anchor'],np.arange(32));messages.append(probs.ravel())
            training_seconds+=time.perf_counter()-begin;local[uid]=model
            record(metrics,folder/'predictions',model,name,ds,c,'local',r)
            print('CLIENT',index+1,'/',len(ids),uid,flush=True)
        result,comm=aggregate(messages,counts,use_sa,kind,cfg['bits'],cfg['threshold'],cfg['fedavg_public_bound'])
        if kind=='Fedavg':unpack(global_model,result)
        else:
            consensus=torch.tensor(result.reshape(32,2),dtype=torch.float32)
            # Small quantization mass error is retained; identical loss in the paired conditions.
            torch.manual_seed(seed+100+r);global_model.train()
            opt=torch.optim.AdamW(global_model.parameters(),lr=.001,weight_decay=.0001)
            begin=time.perf_counter()
            for _ in range(cfg['server_steps']):
                opt.zero_grad(set_to_none=True)
                loss=sum(source_weights[ds]*-(consensus*F.log_softmax(soft.forward(global_model,'global_student',ds,d['anchor'],np.arange(32)),dim=1)).sum(1).mean() for ds,d in domains.items())
                assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(global_model.parameters(),5);opt.step()
            comm['student_training_seconds']=time.perf_counter()-begin
        after=model_hash(global_model);save_model(folder/f'global_round_{r}.pt',global_model)
        # Audit exact broadcast serialization without retaining N redundant files per round.
        for uid in ids:
            replica=copy.deepcopy(global_model);assert model_hash(replica)==after
        audit.append(dict(round=r,input_global_hash=before,output_global_hash=after,recipients=ids,replicas_verified=True))
        for ds,d in domains.items():
            for c in d['clients'].values():record(metrics,folder/'predictions',global_model,'global_student',ds,c,'global',r)
        state_bytes=4*len(pack(global_model));broadcast=len(ids)*state_bytes*(2 if r==cfg['rounds'] else 1)
        communication.append(dict(round=r,local_compute_seconds=training_seconds,model_state_bytes=state_bytes,
            broadcast_bytes=broadcast,download_bytes=broadcast+comm['sa_download_bytes'],
            total_bytes=comm['upload_bytes']+broadcast+comm['sa_download_bytes'],**comm))
        pd.DataFrame(metrics).to_json(folder/'metrics.json',orient='records',double_precision=15)
        pd.DataFrame(communication).to_json(folder/'communication.json',orient='records',double_precision=15)
        dump(folder/'broadcast_audit.json',audit)
        for uid,model in local.items():save_model(folder/'locals'/f'{uid.replace(":","_")}.pt',model)
        print('ROUND COMPLETE',kind,condition,seed,r,flush=True)
        if settings is None:
            from support.sa_communication_report import refresh
            refresh()
    dump(meta,dict(status='complete',signature=token,settings=cfg,kind=kind,condition=condition,seed=seed,
        initial_data_registry=sample_registry(domains),source_weights=source_weights,final_global_hash=after,
        elapsed_seconds=time.perf_counter()-start,test_used=False))
    validate_run(folder,domains)

def validate_run(folder,domains):
    meta=json.loads((folder/'run.json').read_text());assert meta['status']=='complete' and not meta['test_used']
    assert meta['initial_data_registry']==sample_registry(domains)
    clients={c['uid']:c for d in domains.values() for c in d['clients'].values()}
    frame=pd.read_json(folder/'metrics.json');rounds=meta['settings']['rounds']
    ntrain=sum(c['split']=='train' for c in clients.values());assert len(frame)==rounds*(ntrain+len(clients))
    assert not frame.duplicated(['round','client','role']).any()
    for row in frame.to_dict('records'):
        c=clients[row['client']]
        assert row['fit_examples']==len(c['fit_indices'])
        with np.load(folder/'predictions'/row['prediction_file']) as z:
            assert np.array_equal(z['sample_ids'],c['sample_ids'][c['hold_indices']])
            assert not set(z['sample_ids'])&set(c['sample_ids'][c['fit_indices']])
            values,_=evaluate(z['y'],z['p'],row['dataset'])
            for metric in METRICS:assert np.isclose(values[metric],row[metric],atol=1e-10,equal_nan=True)
    global_model=soft.GlobalCNN1D({ds:len(d['config']['feature_names']) for ds,d in domains.items()},meta['seed'])
    initial_hash=model_hash(global_model)
    global_model.load_state_dict(torch.load(folder/f'global_round_{rounds}.pt',weights_only=True,map_location='cpu'))
    assert model_hash(global_model)==meta['final_global_hash']
    for row in frame[(frame['round']==rounds)&(frame.role=='global')].to_dict('records'):
        c=clients[row['client']]
        with np.load(folder/'predictions'/row['prediction_file']) as z:
            assert np.allclose(z['p'],soft.predictions(global_model,'global_student',row['dataset'],c,c['hold_indices']),atol=1e-6)
    for row in frame[(frame['round']==rounds)&(frame.role=='local')].to_dict('records'):
        c=clients[row['client']];ds=row['dataset']
        if meta['kind']=='Fedavg':model=copy.deepcopy(global_model);name='global_student'
        else:
            name=TEACHERS[ds];model=learning.factory(name,c['F'].shape[1],c['X'].shape[1],meta['seed'])
        model.load_state_dict(torch.load(folder/'locals'/f'{row["client"].replace(":","_")}.pt',weights_only=True,map_location='cpu'))
        with np.load(folder/'predictions'/row['prediction_file']) as z:
            assert np.allclose(z['p'],soft.predictions(model,name,ds,c,c['hold_indices']),atol=1e-6)
    comm=pd.read_json(folder/'communication.json');assert len(comm)==rounds
    assert np.allclose(comm.total_bytes,comm.upload_bytes+comm.download_bytes)
    if meta['condition']=='SA':assert comm.secure_integer_sum_verified.all()
    audits=json.loads((folder/'broadcast_audit.json').read_text());assert len(audits)==rounds
    assert audits[0]['input_global_hash']==initial_hash
    for i,a in enumerate(audits):
        assert a['replicas_verified'] and len(a['recipients'])==ntrain
        if i:assert a['input_global_hash']==audits[i-1]['output_global_hash']
    dump(folder/'validation.json',dict(status='complete',metric_rows=len(frame),sample_ids_verified=True,
        global_predictions_recomputed=True,communication_verified=True,signature=meta['signature']))

def run():
    freeze();BASE.mkdir(exist_ok=True)
    with (BASE/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        try:
            data=prepare()
            from support.sa_communication_report import refresh
            refresh()
            for seed in DESIGN['seeds']:
                for kind in ['Fedavg','Softlabels']:
                    for sa in [False,True]:
                        run_one(data,kind,sa,seed);refresh()
            dump(BASE/'status.json',dict(status='complete',validated_runs=20,signature=signature()));refresh()
        except Exception as error:
            dump(BASE/'status.json',dict(status='failed',error=str(error),signature=signature()))
            traceback.print_exc();raise

if __name__=='__main__':run()
