"""Controlled communication ablation: per-anchor probabilities vs MC-dropout mean/std."""
import ast,fcntl,hashlib,json,sys,time,traceback,types
from pathlib import Path
import numpy as np
import torch
from support import sa_communication as core
from support.federated_global.shared import ROOT,dump,sample_registry

BASE=ROOT/'SoftLabels tests/moments_comparison'
MODES=['probabilities64','anomaly32','mc_mean32','mc_meanstd64']
DESIGN=dict(version=2,mc_passes=20,seeds=list(range(2031,2036)),rounds=3,local_epochs=10,server_steps=3,
    models=core.TEACHERS,student=core.DESIGN['global_model'],modes=MODES,sa_enabled=False,
    weights='fit examples / total fit examples; 72.73% Icentia, 27.27% PhysioNet',
    moments='per-anchor population mean/std of anomaly probabilities over 20 MC-dropout passes; BatchNorm stays eval',
    aggregation='mu=sum(w*mu_i); second=sum(w*(sigma_i**2+mu_i**2)); sigma=sqrt(second-mu**2)',
    moments_loss='per-anchor soft CE on [1-mu,mu], weighted by normalized 1/(1+4*mixture_variance); source fit shares',
    moments_limitation='anchor correspondence retained; mixture variance includes within-teacher variability and between-teacher disagreement; not calibrated confidence or privacy',
    reference_loss='per-anchor soft cross entropy',reference_reuse='validated compatible Communication/Softlabels/plain runs, if present',
    primary='fit-weighted local AP on own-client holdouts; last round; five paired seeds',
    secondary='global AP on same holdouts, dev, bytes and local compute time',test_used=False)

def signature():
    return hashlib.sha256(Path(__file__).read_bytes()+core.signature().encode()+json.dumps(DESIGN,sort_keys=True).encode()).hexdigest()

def freeze():
    BASE.mkdir(parents=True,exist_ok=True);path=BASE/'design.json'
    if path.exists():assert json.loads(path.read_text())==DESIGN
    else:dump(path,DESIGN)

def mc_predictions(model,name,ds,anchor,seed,passes=20):
    """Only dropout is stochastic: never update BatchNorm from synthetic anchors."""
    modes={module:module.training for module in model.modules()}
    dropout=[m for m in model.modules() if isinstance(m,(torch.nn.Dropout,torch.nn.Dropout1d,torch.nn.Dropout2d,torch.nn.Dropout3d,torch.nn.AlphaDropout))]
    assert dropout and passes>=2
    model.eval()
    for module in dropout:module.train()
    try:
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.manual_seed(seed)
            values=torch.stack([core.soft.forward(model,name,ds,anchor,np.arange(32)).softmax(1)[:,1] for _ in range(passes)])
        mean=values.mean(0).numpy();std=values.std(0,unbiased=False).numpy()
        return np.column_stack([1-mean,mean]),std
    finally:
        for module,training in modes.items():module.training=training


def encode(probs,mode,std=None):
    p=np.asarray(probs)
    assert p.shape==(32,2) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    if mode=='probabilities64':return p.ravel()
    if mode in ['anomaly32','mc_mean32']:return p[:,1].copy()
    if mode=='mc_meanstd64':
        std=np.asarray(std);assert std.shape==(32,) and np.isfinite(std).all() and (std>=0).all()
        return np.column_stack([p[:,1],std]).ravel()
    raise ValueError(mode)


def aggregate(messages,counts,mode,folder=None,round_=None):
    start=time.perf_counter();x=np.asarray(messages,dtype=float);w=np.asarray(counts,dtype=float);w/=w.sum()
    expected={'probabilities64':64,'anomaly32':32,'mc_mean32':32,'mc_meanstd64':64}[mode]
    assert x.shape==(len(w),expected)
    if mode=='mc_meanstd64':
        pairs=x.reshape(len(w),32,2);mu=np.average(pairs[:,:,0],axis=0,weights=w)
        second=np.average(pairs[:,:,1]**2+pairs[:,:,0]**2,axis=0,weights=w)
        result=np.column_stack([mu,np.sqrt(np.maximum(second-mu*mu,0))]).ravel()
    else:
        result=np.average(x,axis=0,weights=w)
        if mode in ['anomaly32','mc_mean32']:result=np.column_stack([1-result,result]).ravel()
    if folder is not None:
        np.savez_compressed(folder/f'anchor_round_{round_}.npz',messages=x,counts=counts,weights=w,target=result,anchor_indices=np.arange(32))
    upload=4*x.size+4*len(x)
    return result,dict(upload_bytes=upload,sa_download_bytes=0,baseline_upload_bytes=upload,
        aggregation_seconds=time.perf_counter()-start,server_unmask_seconds=0.,aggregate_mae=0.,aggregate_max_error=0.,
        secure_integer_sum_verified=False,values_per_client=expected)


def student_loss(model,ds,anchor,target,mode):
    out=core.soft.forward(model,'global_student',ds,anchor,np.arange(32))
    if mode!='mc_meanstd64':return -(target.reshape(32,2)*torch.nn.functional.log_softmax(out,dim=1)).sum(1).mean()
    moments=target.reshape(32,2);mu=moments[:,0];variance=moments[:,1].square()
    labels=torch.stack([1-mu,mu],dim=1)
    reliability=1/(1+4*variance);reliability=reliability/reliability.mean()
    return (reliability*-(labels*torch.nn.functional.log_softmax(out,dim=1)).sum(1)).mean()


def engine(mode,root=None):
    """Isolate only the payload, its aggregation and the server loss; reuse audited training."""
    source=Path(core.__file__).read_text();tree=ast.parse(source)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_one')
    text=ast.unparse(function)
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
    replace("folder = base / kind / 'artifacts' / str(seed) / condition", "folder = base / 'artifacts' / str(seed)")
    if mode.startswith('mc_'):
        prediction_nodes=[n for n in ast.walk(function) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='probs' for t in n.targets)]
        assert len(prediction_nodes)==1
        old_prediction=ast.unparse(prediction_nodes[0])
        replace(old_prediction,"probs, deviations = mc_predictions(model, name, ds, domains[ds]['anchor'], seed=seed*100000+r*1000+int.from_bytes(hashlib.sha256(uid.encode()).digest()[:2], 'little'), passes=MC_PASSES)")
        replace('messages.append(probs.ravel())','messages.append(encode(probs, MODE, deviations))')
    else:replace('messages.append(probs.ravel())','messages.append(encode(probs, MODE))')
    replace("aggregate(messages, counts, use_sa, kind, cfg['bits'], cfg['threshold'], cfg['fedavg_public_bound'])",
            'aggregate_moments(messages, counts, MODE, folder=folder, round_=r)')
    replace('consensus = torch.tensor(result.reshape(32, 2), dtype=torch.float32)',
            'consensus = torch.tensor(result, dtype=torch.float32)')
    loss_nodes=[n for n in ast.walk(function) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='loss' for t in n.targets)]
    assert len(loss_nodes)==1
    old=ast.unparse(loss_nodes[0].value)
    replace(old,"sum(source_weights[ds]*student_loss(global_model,ds,d['anchor'],consensus,MODE) for ds,d in domains.items())")
    replace('if settings is None:', 'if False:')
    module=types.ModuleType('_moments_'+mode);module.__dict__.update(core.__dict__)
    module.__dict__.update(MODE=mode,BASE=BASE/mode if root is None else Path(root),signature=signature,
        encode=encode,aggregate_moments=aggregate,student_loss=student_loss,mc_predictions=mc_predictions,MC_PASSES=DESIGN['mc_passes'])
    exec(compile(text,str(Path(__file__)),'exec'),module.__dict__)
    return module

def reference(seed):
    folder=core.BASE/'Softlabels/artifacts'/str(seed)/'plain';meta=folder/'run.json'
    if not meta.exists():return None
    d=json.loads(meta.read_text())
    if d.get('status')=='complete' and d.get('signature')==core.signature() and d.get('settings')==core.DESIGN:
        return folder
    return None

def folder_for(mode,seed):
    if mode=='probabilities64':
        p=reference(seed)
        if p is not None:return p
    return BASE/mode/'artifacts'/str(seed)

def complete(mode,seed):
    folder=folder_for(mode,seed);p=folder/'run.json';v=folder/'validation.json'
    if not p.exists() or not v.exists():return False
    d=json.loads(p.read_text());validation=json.loads(v.read_text())
    expected=core.signature() if mode=='probabilities64' and folder==reference(seed) else signature()
    return d.get('status')=='complete' and d.get('signature')==expected and validation.get('signature')==expected

def run():
    freeze()
    with (BASE/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        try:
            domains=core.prepare();dump(BASE/'sample_registry.json',sample_registry(domains));usage=[]
            # Reuse only baseline runs with matching code/design and recomputed validation.
            for seed in DESIGN['seeds']:
                p=reference(seed)
                if p is not None:
                    core.validate_run(p,domains);usage.append(dict(seed=seed,path=str(p.relative_to(ROOT)),signature=core.signature()))
            dump(BASE/'reference_reuse.json',usage)
            from support.softlabels_moments_report import refresh
            refresh()
            for seed in DESIGN['seeds']:
                for mode in MODES:
                    if mode=='probabilities64' and reference(seed) is not None:continue
                    dump(BASE/'status.json',dict(status='running',mode=mode,seed=seed,signature=signature()))
                    cfg=dict(core.DESIGN,representation=mode,mc_passes=DESIGN['mc_passes'] if mode.startswith('mc_') else 0,
                        uncertainty_weighting='normalized 1/(1+4*mixture_variance)' if mode=='mc_meanstd64' else 'none')
                    module=engine(mode)
                    module.run_one(domains,'Softlabels',False,seed,settings=cfg,base=BASE/mode)
                    refresh()
            assert all(complete(m,s) for m in MODES for s in DESIGN['seeds'])
            dump(BASE/'status.json',dict(status='complete',validated_conditions=20,reused_baselines=len(usage),signature=signature()));refresh()
        except Exception as error:
            dump(BASE/'status.json',dict(status='failed',error=str(error),signature=signature()));traceback.print_exc();raise

if __name__=='__main__':run()
