"""Config-A training with local logit LDP and real holdout AP/recall/FAR."""
import ast,fcntl,hashlib,json,sys,traceback,types
from pathlib import Path
import diffprivlib
import numpy as np
import pandas as pd
import torch
from . import training_base as core
from .mechanism import ClientMechanism,Ledger
from .experiencias_logits import SAIDA_PADRAO
from .ldp_logits import softmax_binario
from support.federated_global.shared import ROOT,dump,sample_registry

FOLDER=ROOT/'Privacy Techniques/LDP'
BASE=ROOT/'artifacts/ldp_logits/closed_loop'
COMPAT=Path(__file__).with_name('baseline_compatibility.json')

def design():
    path=BASE/'design.json'
    if path.exists():return json.loads(path.read_text())
    frame=pd.read_csv(SAIDA_PADRAO/'top3_por_epsilon.csv');selected=frame[frame.epsilon_total==4].head(3)
    assert len(selected)==3
    profiles=[dict(id=f'P{i+1}',mechanism=r.mechanism,representation=r.representation,clip_C=float(r.clip_C)) for i,r in enumerate(selected.itertuples())]
    cfg=dict(version=1,epsilon_total=4.,delta_total_gaussian=1e-5,seeds=list(range(2031,2036)),rounds=3,local_epochs=10,server_steps=3,
        models=core.TEACHERS,student=core.DESIGN['global_model'],profiles=profiles,selection='offline consensus MAE at epsilon=4; exploratory, no clinical labels',
        fit_counts={'icentia11k':2048,'physionet2017':768},weighting='fit / total fit',sa=False,test_used=False,
        privacy_unit='replace a complete prepared client message; fixed public participant set and fixed transforms',
        limitations='raw data end-to-end privacy not established; preprocessing used pooled fit statistics; evaluation and parameter selection are not DP releases',
        composition='epsilon/delta divided by three rounds; sum all private attempts and seeds in ledger; controls unprotected')
    dump(path,cfg);return cfg

def signature():
    files=[Path(__file__),Path(__file__).with_name('mechanism.py'),Path(__file__).with_name('ldp_logits.py'),COMPAT]
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)+core.signature().encode()+json.dumps(design(),sort_keys=True).encode()+diffprivlib.__version__.encode()).hexdigest()

def conditions():
    result=[dict(name='baseline_raw',mode='raw',mechanism=None,representation='original',clip_C=None,epsilon_total=None,delta_total=None)]
    for profile in design()['profiles']:
        for mode in ['clipped','ldp']:
            result.append(dict(profile,name=profile['id']+'_'+mode,mode=mode,epsilon_total=4. if mode=='ldp' else None,
                delta_total=(1e-5 if profile['mechanism']=='gaussiano' else 0.) if mode=='ldp' else None))
    return result

def reference(seed):
    info=json.loads(COMPAT.read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in info['dependency_sha256'].items())
    path=ROOT/'Privacy Techniques/SA/Communication/Softlabels/artifacts'/str(seed)/'plain'
    if not (path/'run.json').exists():return None
    d=json.loads((path/'run.json').read_text())
    if d.get('status')=='complete' and d.get('signature')==info['reference_signature'] and d.get('settings')==core.DESIGN:return path
    return None

def folder_for(condition,seed):
    if condition['mode']=='raw':
        existing=reference(seed)
        if existing is not None:return existing
    return BASE/'runs'/condition['name']/'artifacts'/str(seed)

def complete(condition,seed):
    path=folder_for(condition,seed);meta=path/'run.json';v=path/'validation.json'
    if not meta.exists() or not v.exists():return False
    d=json.loads(meta.read_text());report=json.loads(v.read_text())
    expected=json.loads(COMPAT.read_text())['reference_signature'] if condition['mode']=='raw' and path==reference(seed) else signature()
    return d.get('status')=='complete' and d.get('signature')==expected==report.get('signature')

def engine(condition,ledger,attempt,root):
    tree=ast.parse(Path(core.__file__).read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_one')
    text=ast.unparse(fn)
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
    replace("condition = 'SA' if use_sa else 'plain'", "condition = LDP_CONDITION['name']")
    replace("folder = base / kind / 'artifacts' / str(seed) / condition","folder = base / 'artifacts' / str(seed)")
    replace('if settings is None:','if False:')
    if condition['mode']!='raw':
        prediction=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='probs' for t in n.targets))
        replace(ast.unparse(prediction),"probs = private_prediction(model,name,ds,domains[ds]['anchor'],uid,r)")
    module=types.ModuleType('_ldp_'+condition['name']);module.__dict__.update(core.__dict__)
    mechanisms={}
    def private_prediction(model,name,ds,anchor,uid,round_):
        if uid not in mechanisms:mechanisms[uid]=ClientMechanism(condition,design()['rounds'],ledger,attempt,uid)
        mechanism=mechanisms[uid]
        z=core.soft.predictions(model,name,ds,anchor,np.arange(32),as_logits=True)
        assert mechanism.releases==round_-1
        protected=mechanism.release(z)
        dump(Path(root)/'certificates'/f'{uid.replace(":","_")}.json',mechanism.certificate())
        return softmax_binario(protected).astype(np.float32)
    module.__dict__.update(BASE=Path(root),signature=signature,private_prediction=private_prediction,LDP_CONDITION=condition)
    exec(compile(text,str(Path(__file__)),'exec'),module.__dict__)
    return module,mechanisms

def validate(condition,seed,domains):
    path=folder_for(condition,seed);core.validate_run(path,domains)
    if condition['mode']!='raw':
        train=[c for d in domains.values() for c in d['clients'].values() if c['split']=='train']
        certificates=list((path/'certificates').glob('*.json'))
        assert len(certificates)==len(train)
        for p in certificates:
            d=json.loads(p.read_text());assert d['releases']==design()['rounds']
            if condition['mode']=='ldp':assert d['secure_randomness'] and d['epsilon_cycle']==4.
    return path

def run():
    cfg=design();BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        try:
            dump(BASE/'status.json',dict(status='preparing',stage='fixed data and paired baselines',signature=signature()))
            domains=core.prepare();dump(BASE/'sample_registry.json',sample_registry(domains))
            reuse=[]
            for seed in cfg['seeds']:
                p=reference(seed)
                if p is not None:core.validate_run(p,domains);reuse.append(dict(seed=seed,path=str(p.relative_to(ROOT))))
            dump(BASE/'reference_reuse.json',reuse)
            ledger=Ledger(BASE/'privacy_ledger.json')
            ledger.data['unprotected_reference_reused']=bool(reuse);dump(ledger.path,ledger.data)
            from .notebooks import refresh
            refresh()
            for seed in cfg['seeds']:
                for condition in conditions():
                    if complete(condition,seed):continue
                    dump(BASE/'status.json',dict(status='running',condition=condition['name'],seed=seed,signature=signature()))
                    attempt=ledger.begin(f"{condition['name']}/{seed}")
                    base=BASE/'runs'/condition['name'];folder=base/'artifacts'/str(seed)
                    module,mechanisms=engine(condition,ledger,attempt,folder)
                    settings=dict(core.DESIGN,ldp_condition=condition,sa_enabled=False)
                    module.run_one(domains,'Softlabels',False,seed,settings=settings,base=base)
                    validate(condition,seed,domains);ledger.finish(attempt);refresh()
            assert all(complete(c,s) for c in conditions() for s in cfg['seeds'])
            dump(BASE/'status.json',dict(status='complete',validated_conditions=35,reused_baselines=len(reuse),signature=signature()));refresh()
        except Exception as error:
            dump(BASE/'status.json',dict(status='failed',error=str(error),signature=signature()));traceback.print_exc();raise

if __name__=='__main__':run()
