"""Frozen P2 LDP + SA training; final unseen-client test only after all seeds."""
import fcntl,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from support.ldp_logits import closed_loop as previous
from support.ldp_logits.mechanism import Ledger
from support.ldp_logits import training_base as core
from support.federated_global.shared import ROOT,dump
from . import protocol as SA

BASE=ROOT/'artifacts/ldp_sa'
FOLDER=ROOT/'Privacy Techniques/LDP + SA'
CONDITION=dict(name='P2_LDP_SA',mode='ldp',mechanism='gaussiano',representation='margem',clip_C=.5,epsilon_total=4.,delta_total=1e-5)
DESIGN=dict(version=1,seeds=list(range(2031,2036)),rounds=3,local_epochs=10,server_steps=3,
    condition=CONDITION,bits=24,threshold=12,sa_mode='one fit-weighted sum',dropouts_in_training=False,
    data='same grouped training data as completed LDP study',source_weights={'icentia11k':16/22,'physionet2017':6/22},
    selection='P2 chosen by completed local holdout AP before opening test; not independently validated selection',
    model_selection_test_used=False,primary_local='fit-weighted own-client holdout AP at fixed round 3',
    final_test='global CNN1D, all eligible prepared test examples, unseen clients, all five seeds, no adaptation',
    test_threshold='argmax / 0.5 fixed, no tuning',privacy_scope='prepared client message with fixed transforms; not raw ECG pipeline',
    malicious_security='LDP protects honest clients conditional on the specified channel; SA server is semi-honest; no Byzantine/poisoning defense',
    randomness='secure non-seeded LDP noise and fresh SA keys every round; training seeds only control model training',
    noise_reduction=False,privacy_composition='epsilon=4 and delta=1e-5 per client cycle; sum seeds/attempts; separate historical ledger also applies')

def signature():
    files=[Path(__file__),Path(__file__).with_name('test_evaluation.py'),Path(SA.__file__),
           Path(previous.__file__),Path(core.__file__),ROOT/'support/ldp_logits/mechanism.py',
           ROOT/'support/ldp_logits/ldp_logits.py',ROOT/'support/federated_global/soft.py',
           ROOT/'support/federated_global/personalized.py',ROOT/'SoftLabels tests/common/exp4_data.py',
           ROOT/'SoftLabels tests/common/data.py']+[ROOT/'FedAvg tests/common'/n for n in ['models.py','learning.py','metrics.py']]
    return hashlib.sha256(json.dumps(DESIGN,sort_keys=True).encode()+b''.join(p.read_bytes() for p in files)).hexdigest()

def freeze():
    BASE.mkdir(parents=True,exist_ok=True);FOLDER.mkdir(parents=True,exist_ok=True)
    config=dict(design=DESIGN,signature=signature())
    path=BASE/'frozen_design.json'
    if path.exists():assert json.loads(path.read_text())==config,'Frozen experiment changed; create a new version'
    else:dump(path,config)
    return config

def aggregate(messages,counts,use_sa,kind,bits=24,threshold=12,bound=64.):
    assert use_sa and kind=='Softlabels' and bits==24 and threshold==12
    x=np.asarray(messages,dtype=np.float64);counts=np.asarray(counts,dtype=float)
    if x.ndim!=2 or x.shape[1]!=64 or len(x)<threshold or counts.shape!=(len(x),):
        raise ValueError('Invalid cohort or dimensions')
    if not np.isfinite(x).all() or (x<0).any() or (x>1).any() or not np.allclose(x.reshape(-1,32,2).sum(-1),1,atol=1e-6):
        raise ValueError('Invalid probability message')
    if not np.isfinite(counts).all() or (counts<=0).any():raise ValueError('Invalid public fit counts')
    w=counts/counts.sum();start=time.perf_counter()
    vectors={i+1:SA.quantize(w[i]*x[i],bits) for i in range(len(x))}
    integer_sum,report=SA.secure_sum(vectors,threshold)
    # Trusted simulator audit, not information available to a deployed server.
    assert np.array_equal(integer_sum,sum(vectors.values())%SA.MOD)
    result=SA.dequantize(integer_sum,bits)
    reference=np.average(x,axis=0,weights=w)
    max_error=float(np.max(abs(result-reference)))
    assert max_error<=len(x)*.5/(1<<bits)+1e-12
    upload=4*len(x)+int(round(report['upload_bytes_mean']*len(x)))
    download=int(round(report['download_bytes_mean']*len(x)))
    return result,dict(upload_bytes=upload,sa_download_bytes=download,baseline_upload_bytes=4*x.size+4*len(x),
        aggregation_seconds=time.perf_counter()-start,server_unmask_seconds=report['server_seconds'],
        aggregate_mae=float(abs(result-reference).mean()),aggregate_max_error=max_error,
        secure_integer_sum_verified=True,values_per_client=x.shape[1])

def folder_for(seed):return BASE/'runs/artifacts'/str(seed)

def validate(seed,domains):
    folder=folder_for(seed);core.validate_run(folder,domains)
    meta=json.loads((folder/'run.json').read_text());assert meta['signature']==signature()
    assert meta['settings']['sa_enabled'] and meta['condition']==CONDITION['name']
    import pandas as pd
    comm=pd.read_json(folder/'communication.json');assert comm.secure_integer_sum_verified.all()
    assert (comm.aggregate_max_error<=22*.5/(1<<24)+1e-12).all()
    certificates=list((folder/'certificates').glob('*.json'));assert len(certificates)==22
    for file in certificates:
        cert=json.loads(file.read_text())
        assert cert['releases']==3 and cert['secure_randomness'] and cert['epsilon_cycle']==4. and cert['delta_cycle']==1e-5
    return folder

def run():
    freeze()
    with (BASE/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        from .notebook import refresh
        try:
            dump(BASE/'status.json',dict(status='preparing',signature=signature()));refresh()
            # prepare() only reads train/dev caches. Test arrays are opened after all training.
            domains=core.prepare()
            dump(BASE/'sample_registry.json',core.sample_registry(domains))
            ledger=Ledger(BASE/'privacy_ledger.json')
            for seed in DESIGN['seeds']:
                folder=folder_for(seed);meta=folder/'run.json'
                if meta.exists() and json.loads(meta.read_text()).get('status')=='complete':
                    validate(seed,domains);continue
                dump(BASE/'status.json',dict(status='training',seed=seed,signature=signature()));refresh()
                attempt=ledger.begin(f'P2_LDP_SA/{seed}')
                module,_=previous.engine(CONDITION,ledger,attempt,folder)
                module.signature=signature;module.aggregate=aggregate
                settings=dict(core.DESIGN,ldp_condition=CONDITION,sa_enabled=True)
                module.run_one(domains,'Softlabels',True,seed,settings=settings,base=BASE/'runs')
                validate(seed,domains);ledger.finish(attempt);refresh()
            # Freeze and validate ALL final checkpoints before inspecting held-out test arrays.
            for seed in DESIGN['seeds']:validate(seed,domains)
            dump(BASE/'test_gate.json',dict(training_complete=True,configuration_frozen=True,
                signature=signature(),seeds=DESIGN['seeds'],round=3,selection_on_test=False))
            dump(BASE/'status.json',dict(status='evaluating_test',signature=signature()));refresh()
            from .test_evaluation import evaluate_test
            evaluate_test(domains)
            dump(BASE/'status.json',dict(status='complete',validated_training_runs=5,test_complete=True,signature=signature()));refresh()
        except Exception as error:
            dump(BASE/'status.json',dict(status='failed',error=str(error),signature=signature()));refresh()
            traceback.print_exc();raise

if __name__=='__main__':run()
