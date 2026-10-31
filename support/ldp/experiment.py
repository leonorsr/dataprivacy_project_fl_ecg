"""Closed-loop LDP experiment; no changes to the concurrently running final comparison."""
import contextlib,fcntl,hashlib,importlib.util,json,shutil,sys,uuid
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from support import final_candidates as final
from support.federated_global import shared
from support.federated_global.shared import ROOT,DOMAINS,dump
from .mechanism import LocalLogitMechanism,PrivacyLedger
from .data import prepare

FOLDER=ROOT/'Privacy Techniques/LDP'
BASE=FOLDER/'artifacts'
CONFIG=FOLDER/'config.json'


def settings():
    cfg=json.loads(CONFIG.read_text())
    assert cfg['privacy_unit']=='client_replacement' and cfg['rounds']==3
    assert cfg['local_epochs_per_round']==final.LOCAL_EPOCHS
    assert cfg['clip_l1_bound']>0 and len(set(cfg['training_seeds']))==len(cfg['training_seeds'])
    assert cfg['epsilon_values'] and all(np.isfinite(v) and v>0 for v in cfg['epsilon_values'])
    return cfg


def winner():
    path=final.FOLDER/'decision.json';validation=final.BASE/'validation_report.json'
    if not path.exists() or not validation.exists():
        raise RuntimeError('A candidata final ainda não foi decidida e validada. Executar sem --run para consultar o plano; piloto sintético em support/model_tests/verify_ldp.py.')
    chosen=json.loads(path.read_text());report=json.loads(validation.read_text())
    assert chosen['signature']==final.signature() and report['status']=='complete'
    assert chosen['status']=='selected_after_validation'
    amendment=final.FOLDER/'analysis_amendment.json'
    if amendment.exists():
        from support.local_decision import signature as analysis_signature
        if chosen.get('analysis_priority')!='local_teacher_dev' or chosen.get('analysis_signature')!=analysis_signature():
            raise RuntimeError('A decisão pelo PR-AUC local em dev ainda não foi finalizada; aguardar o pós-processamento da análise.')
    return chosen


def signature():
    import diffprivlib
    files=sorted(Path(__file__).parent.glob('*.py'))
    extra=b''
    for p in [final.FOLDER/'analysis_amendment.json',final.FOLDER/'decision.json']:
        if p.exists():extra+=p.read_bytes()
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)+CONFIG.read_bytes()+final.signature().encode()+diffprivlib.__version__.encode()+extra).hexdigest()


def conditions(cfg):
    return [dict(name='baseline_raw',mode='raw',epsilon=None),dict(name='baseline_clipped',mode='clipped',epsilon=None)]+[
        dict(name='epsilon_'+str(e).replace('.','p'),mode='ldp',epsilon=e) for e in cfg['epsilon_values']]


def build_engine(condition,ledger,attempt,cfg):
    engine=final.training_engine()
    engine.SOFT=BASE;engine.signature=signature
    Parent=engine.TeacherClient
    class PrivateTeacher(Parent):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert len(self.c['fit_indices'])==cfg['public_fit_per_train_client'],'Fit cardinality must match the public design'
            self.mechanism=LocalLogitMechanism(condition['mode'],condition['epsilon'],cfg['clip_l1_bound'],cfg['rounds'],ledger,attempt,self.c['uid'])
        def fit(self,parameters,config):
            arrays,n_examples,metrics=super().fit(parameters,config)
            assert len(arrays)==1
            return [self.mechanism.release(arrays[0])],n_examples,metrics
    ParentStrategy=engine.ServerDistillation
    class PrivateStrategy(ParentStrategy):
        def aggregate_fit(self,server_round,results,failures):
            parameters,metrics=super().aggregate_fit(server_round,results,failures)
            audit=self.audit[-1]
            audit.update(ldp_mode=condition['mode'],epsilon_client_cycle=condition['epsilon'],
                epsilon_client_round=condition['epsilon']/cfg['rounds'] if condition['mode']=='ldp' else None,
                joint_sensitivity=2*cfg['clip_l1_bound'] if condition['mode']=='ldp' else None,
                privacy_unit='replace all training records of a known fixed-size participant',
                noise_before_transmission=True,payload='locally randomised anchor logits' if condition['mode']=='ldp' else 'non-private baseline logits')
            dump(self.folder/'server_training_audit.json',self.audit)
            # Only public mechanism parameters are released, never private clipping statistics.
            for uid,client in self.clients.items():
                dump(self.folder/'mechanism_certificates'/f'{uid.replace(":","_")}.json',client.mechanism.certificate())
            return parameters,metrics
    engine.TeacherClient=PrivateTeacher;engine.ServerDistillation=PrivateStrategy
    return engine


def move_private_artifacts(folder):
    """Separate teacher states and evaluation labels from the server-export whitelist."""
    private=folder/'research_private';private.mkdir(exist_ok=True)
    if (folder/'teachers').exists():shutil.move(str(folder/'teachers'),str(private/'teacher_states'))
    metrics=folder/'metrics.json'
    if metrics.exists():
        f=pd.read_json(metrics)
        pred=private/'predictions';pred.mkdir(exist_ok=True)
        for filename in f.prediction_file.unique():
            source=folder/filename
            if source.exists() and source.parent==folder:
                shutil.move(str(source),str(pred/source.name))
        f['prediction_file']=f.prediction_file.map(lambda s:'research_private/predictions/'+Path(s).name)
        f.to_json(metrics,orient='records',double_precision=15)


def collect():
    frames=[]
    for path in (BASE/'runs').glob('*/*/run_config.json'):
        cfg=json.loads(path.read_text())
        if cfg.get('status')!='complete' or cfg.get('signature')!=signature():continue
        f=pd.read_json(path.with_name('metrics.json'));f['condition']=cfg['name'];f['mode']=cfg['ldp_mode'];f['epsilon']=cfg.get('epsilon_client_cycle');f['seed']=cfg['seed'];frames.append(f)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()


def report():
    f=collect()
    if f.empty:return
    keys=['condition','mode','dataset','split','approach']
    per=f.groupby(keys+['seed'])[['pr_auc_ap','roc_auc','recall','far','f1']].mean().reset_index()
    summary=per.groupby(keys)[['pr_auc_ap','roc_auc','recall','far','f1']].mean().reset_index()
    spread=per.groupby(keys).agg(pr_auc_std=('pr_auc_ap','std'),seeds=('seed','nunique')).reset_index()
    summary=summary.merge(spread,on=keys)
    per.to_json(BASE/'per_seed.json',orient='records',double_precision=15)
    summary.to_json(BASE/'summary.json',orient='records',double_precision=15)
    # Paired training seeds; privacy noise itself is deliberately not public-seeded.
    global_per=per[per.approach.isin(['local_teacher','global_student'])&(per.split=='dev')]
    baseline=global_per[global_per.condition=='baseline_clipped'][['dataset','seed','approach','pr_auc_ap']].rename(columns={'pr_auc_ap':'baseline_ap'})
    paired=global_per.merge(baseline,on=['dataset','seed','approach']);paired['delta_vs_clipped']=paired.pr_auc_ap-paired.baseline_ap
    paired.to_json(BASE/'paired_utility.json',orient='records',double_precision=15)


def validate(data,require_complete=False):
    validator=final.isolate('_ldp_validation','validation.py');validator.SOFT=BASE;validator.signature=signature;validator.soft_domains=lambda _:data
    result=validator.validate_soft(require_complete=False)
    cfg=settings();completed=0
    for meta in (BASE/'runs').glob('*/*/run_config.json'):
        run=json.loads(meta.read_text())
        if run.get('status')!='complete' or run.get('signature')!=signature():continue
        folder=meta.parent;completed+=1
        certificates=list((folder/'mechanism_certificates').glob('*.json'))
        assert len(certificates)==sum(c['split']=='train' for d in data.values() for c in d['clients'].values())
        for path in certificates:
            cert=json.loads(path.read_text());assert cert['releases']==3
            if run['ldp_mode']=='ldp':
                assert cert['secure_randomness'] and cert['delta']==0
                assert np.isclose(cert['epsilon_per_round']*3,run['epsilon_client_cycle'])
                assert np.isclose(cert['joint_l1_sensitivity'],2*cfg['clip_l1_bound'])
                assert np.isclose(cert['laplace_scale'],2*cfg['clip_l1_bound']*3/run['epsilon_client_cycle'])
        assert not (folder/'teachers').exists(),'Private teacher weights must not sit in server exports.'
        run['ldp_validated']=True;dump(meta,run)
    expected=len(conditions(cfg))*len(cfg['training_seeds'])
    if require_complete:assert completed==expected,(completed,expected)
    result.update(status='complete' if completed==expected else 'partial',expected_runs=expected,
        ldp_message_mechanism_verified=True,privacy_scope='one cycle conditioned on fixed public setup; not all research outputs',
        numerical_backend='diffprivlib Laplace; SystemRandom; delta=0')
    dump(BASE/'validation_report.json',result);return result


def plan():
    cfg=settings()
    try:chosen=winner();state='ready_after_explicit_run'
    except RuntimeError:chosen=None;state='waiting_for_final_candidate'
    eps_sum=sum(cfg['epsilon_values'])*len(cfg['training_seeds'])
    result=dict(status=state,settings=cfg,winner=chosen,expected_runs=len(conditions(cfg))*len(cfg['training_seeds']),
        privacy_per_cycle='epsilon_total split equally across three joint-vector releases',
        epsilon_composition_all_private_grid_cycles=eps_sum,
        combined_research_release_privacy='No finite guarantee if raw/clipped baselines, teacher weights or private evaluation labels are also exposed',
        local_noise='cryptographic random_state=None; never publish or reuse noise seeds')
    return result


def run(epsilon=None,baseline=None):
    cfg=settings();chosen=winner()
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Já existe uma execução LDP.')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        design=dict(settings=cfg,winner=chosen,signature=signature(),sources=cfg['sources'])
        frozen=FOLDER/'design.json'
        if frozen.exists():assert json.loads(frozen.read_text())==design,'Changing the frozen design requires a new protocol and preservation/review of the cumulative privacy ledger.'
        else:dump(frozen,design)
        ledger=PrivacyLedger(BASE/'private_privacy_ledger.json')
        data=prepare(BASE,calibration_anchors=cfg['public_calibration_anchors'])
        selected=conditions(cfg)
        if epsilon is not None:selected=[c for c in selected if c['mode']=='ldp' and np.isclose(c['epsilon'],epsilon)]
        if baseline is not None:selected=[c for c in selected if c['mode']==baseline]
        assert selected,'Requested condition is not in the predeclared grid'
        try:
            for seed in cfg['training_seeds']:
                for condition in selected:
                    folder=BASE/'runs'/condition['name']/str(seed);path=folder/'run_config.json'
                    if path.exists():
                        old=json.loads(path.read_text())
                        if old.get('status')=='complete' and old.get('signature')==signature():continue
                    attempt=ledger.begin(f"{condition['name']} seed {seed}",condition['epsilon'])
                    engine=build_engine(condition,ledger,attempt,cfg)
                    dump(BASE/'status.json',dict(status='running',condition=condition,seed=seed))
                    run_cfg=dict(experiment='runs',name=condition['name'],models=chosen['teachers'],assignments={},seed=seed,
                        ldp_mode=condition['mode'],epsilon_client_cycle=condition['epsilon'],local_epochs_per_round=cfg['local_epochs_per_round'],privacy_attempt=attempt)
                    engine.run_pair(run_cfg,data)
                    move_private_artifacts(folder);ledger.finish(attempt);report()
                    from .notebook import refresh
                    refresh()
            result=validate(data,require_complete=epsilon is None and baseline is None)
            dump(BASE/'status.json',dict(status=result['status'],validation_passed=True))
            from .notebook import refresh
            refresh()
        except Exception as error:
            dump(BASE/'status.json',dict(status='failed',error=str(error)));raise


def export_server(condition,seed,destination):
    """Only DP messages and global checkpoints; never raw baselines or private metrics."""
    folder=BASE/'runs'/condition/str(seed);cfg=json.loads((folder/'run_config.json').read_text())
    assert cfg['ldp_mode']=='ldp' and cfg['status']=='complete' and cfg['signature']==signature() and cfg.get('ldp_validated')
    target=Path(destination);target.mkdir(parents=True,exist_ok=True)
    for pattern in ['global_round_*.pt','global_final.pt','teacher_round_*.npz','server_training_audit.json']:
        for path in folder.glob(pattern):shutil.copy2(path,target/path.name)
    dump(target/'privacy_scope.json',dict(epsilon_per_client_cycle=cfg['epsilon_client_cycle'],delta=0,
        note='Multiple exported cycles compose; previous non-private outputs are not retroactively protected. Public setup and known participation conditioned on.',
        raw_teacher_weights_exported=False,private_evaluation_exported=False))
