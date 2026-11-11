"""Predeclared two-candidate confirmation; isolated from previous training signatures."""
import ast,contextlib,fcntl,hashlib,importlib.util,io,json,shutil,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .federated_global import shared,personalized
from .federated_global.shared import ROOT,DOMAINS,dump

FOLDER=ROOT/'SoftLabels tests/final candidates'
BASE=FOLDER/'artifacts'
DESIGN=FOLDER/'design.json'
SEEDS=list(range(2031,2036))
LOCAL_EPOCHS=10
EXPECTED_PAIRS=len(SEEDS)*2
CANDIDATES={
 'A_AE_CNN__MobileNetV2':{'icentia11k':'Autoencoder_CNN_GradCAM','physionet2017':'MobileNetV2'},
 'B_CNN1D__MobileNetV2':{'icentia11k':'CNN1D','physionet2017':'MobileNetV2'}}
METRICS=['pr_auc_ap','roc_auc','recall','far','f1']


def definition():
    return dict(protocol='two_final_distillation_candidates_v1',candidates=CANDIDATES,student='CNN1D',
        training_seeds=SEEDS,rounds=3,local_epochs_per_round=LOCAL_EPOCHS,server_steps_per_round=3,anchors=32,split_seed=2026,
        data='saved grouped exp4 sample; 128 fit per train client after fit+selection refit; 32 holdout per train; up to 128 dev per client',
        train_clients={'icentia11k':16,'physionet2017':6},dev_clients={'icentia11k':2,'physionet2017':1},
        normalization='source mean/scale on fit+selection only; never holdout/dev',test_used=False,
        primary='macro client PR-AUC of global student on dev, mean across five paired training seeds',
        decision='minimize worst-origin AP regret relative to the better candidate in each origin; tie within 1e-6: lower mean regret, then lower total runtime, then name',
        uncertainty='paired seed deltas per origin; not population confidence; dev clients previously explored',
        independent_test_claim=False,source_protocol_signature=shared.signature())


def freeze():
    FOLDER.mkdir(parents=True,exist_ok=True)
    proposed=definition()
    if DESIGN.exists():assert json.loads(DESIGN.read_text())==proposed,'Design changed; archive this confirmation before modifying it.'
    else:dump(DESIGN,proposed)
    return proposed


def signature():
    return hashlib.sha256(Path(__file__).read_bytes()+DESIGN.read_bytes()+shared.signature().encode()).hexdigest()


def isolate(name,filename):
    spec=importlib.util.spec_from_file_location('support.federated_global.'+name,ROOT/'support/federated_global'/filename)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def training_engine():
    # Configure the two supervised fit call sites in an isolated copy of the audited engine.
    # Public KD remains one step per feedback round; private controls keep two extra steps.
    source=ROOT/'support/federated_global/soft.py'
    tree=ast.parse(source.read_text());changed=0
    for node in ast.walk(tree):
        if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                and isinstance(node.func.value,ast.Name) and node.func.value.id=='learning'
                and node.func.attr=='fit' and any(k.arg=='ae_epochs' for k in node.keywords)):
            assert isinstance(node.args[3],ast.Constant) and node.args[3].value==1
            node.args[3]=ast.Constant(LOCAL_EPOCHS);changed+=1
        if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('three supervised epochs plus two extra private epochs'):
            node.value=f'{3*LOCAL_EPOCHS} supervised epochs plus two extra private epochs vs two public-KD steps; FLOPs/examples/time not matched'
    assert changed==2,'Unexpected engine layout: review local/control training before running.'
    ast.fix_missing_locations(tree)
    name='support.federated_global._configured_final_engine'
    spec=importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    exec(compile(tree,str(source),'exec'),module.__dict__)
    module.SOFT=BASE;module.signature=signature
    return module


def prepare():
    freeze()
    spec=importlib.util.spec_from_file_location('global_soft.final_grouped_data',ROOT/'SoftLabels tests/common/exp4_data.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    module.BASE=BASE/'data'
    source=ROOT/'SoftLabels tests/artifacts/exp4/data'
    for p in source.glob('*/*.npz'):
        target=module.BASE/'data'/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():shutil.copy2(p,target)
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(target.read_bytes()).digest()
    raw=module.prepare()
    data=personalized.transform(raw,final=True)
    registry=shared.sample_registry(data)
    fit={v for c in registry.values() for v in c['fit']};hold={v for c in registry.values() for v in c['holdout']}
    assert not fit&hold and all(c['split'] in {'train','dev'} for c in registry.values())
    assert sum(c['split']=='train' for c in registry.values())==22
    dump(BASE/'sample_registry.json',registry)
    counts=[]
    for ds,d in data.items():
        for split in ['train','dev']:
            cs=[c for c in d['clients'].values() if c['split']==split]
            counts.append(dict(dataset=ds,split=split,clients=len(cs),fit=sum(len(c['fit_indices']) for c in cs),evaluation=sum(len(c['hold_indices']) for c in cs)))
    dump(BASE/'sample_counts.json',counts)
    return data


def configurations():
    return [dict(experiment='confirmation',name=name,models=models,assignments={},seed=seed,local_epochs_per_round=LOCAL_EPOCHS)
            for seed in SEEDS for name,models in CANDIDATES.items()]


def collect():
    frames=[];timings=[]
    for cfg in configurations():
        path=BASE/'confirmation'/cfg['name']/str(cfg['seed'])/'run_config.json'
        if not path.exists():continue
        meta=json.loads(path.read_text())
        if meta.get('status')!='complete' or meta.get('signature')!=signature():continue
        frame=pd.read_json(path.with_name('metrics.json'));frame['candidate']=cfg['name'];frame['seed']=cfg['seed'];frames.append(frame)
        timings.append(dict(candidate=cfg['name'],seed=cfg['seed'],elapsed_seconds=meta['elapsed_seconds']))
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()),pd.DataFrame(timings)


def decide(per_seed,timings):
    dev=per_seed[(per_seed.approach=='global_student')&(per_seed.split=='dev')]
    assert dev.groupby(['candidate','dataset']).seed.nunique().eq(len(SEEDS)).all()
    expected={(candidate,ds,seed) for candidate in CANDIDATES for ds in DOMAINS for seed in SEEDS}
    assert set(dev[['candidate','dataset','seed']].itertuples(index=False,name=None))==expected
    assert not dev.duplicated(['candidate','dataset','seed']).any() and np.isfinite(dev.pr_auc_ap).all()
    assert set(dev.candidate)==set(CANDIDATES) and set(dev.dataset)==set(DOMAINS)
    mean=dev.groupby(['candidate','dataset']).pr_auc_ap.mean().unstack('dataset')
    regret=mean.max(axis=0)-mean
    seconds=timings.groupby('candidate').elapsed_seconds.sum()
    ranking=pd.DataFrame(dict(worst_origin_regret=regret.max(axis=1),mean_regret=regret.mean(axis=1),total_seconds=seconds))
    ranking=ranking.sort_values(['worst_origin_regret','mean_regret','total_seconds'])
    # Explicit tolerance for metric ties, with runtime used only after both regrets tie.
    contenders=ranking[ranking.worst_origin_regret<=ranking.worst_origin_regret.min()+1e-6]
    contenders=contenders[contenders.mean_regret<=contenders.mean_regret.min()+1e-6]
    winner=contenders.assign(candidate_name=contenders.index).sort_values(['total_seconds','candidate_name']).index[0]
    a,b=list(CANDIDATES)
    paired=dev.pivot(index=['dataset','seed'],columns='candidate',values='pr_auc_ap');paired['A_minus_B']=paired[a]-paired[b]
    differences=paired.groupby('dataset').A_minus_B.agg(['mean','std','min','max']).reset_index()
    differences['positive_seeds_A']=paired.groupby('dataset').A_minus_B.apply(lambda x:int((x>1e-12).sum())).values
    return winner,mean,ranking,paired,differences


def summarize(final=False):
    frame,timings=collect()
    if frame.empty:return None
    keys=['candidate','dataset','split','approach']
    per_seed=frame.groupby(keys+['seed'])[METRICS].mean().reset_index()
    summary=per_seed.groupby(keys)[METRICS].mean().reset_index()
    spread=per_seed.groupby(keys).agg(pr_auc_std=('pr_auc_ap','std'),seeds=('seed','nunique')).reset_index()
    summary=summary.merge(spread,on=keys)
    per_seed.to_json(BASE/'per_seed.json',orient='records',double_precision=15)
    summary.to_json(BASE/'summary.json',orient='records',double_precision=15)
    timings.to_json(BASE/'timings.json',orient='records',double_precision=15)
    if not final:return None
    assert len(timings)==EXPECTED_PAIRS
    winner,mean,ranking,paired,differences=decide(per_seed,timings)
    ranking.reset_index(names='candidate').to_json(BASE/'decision_scores.json',orient='records',double_precision=15)
    differences.to_json(BASE/'paired_differences.json',orient='records',double_precision=15)
    choice=dict(status='selected_after_validation',winner=winner,teachers=CANDIDATES[winner],student='CNN1D',
        criterion=definition()['decision'],signature=signature(),paired_runs=EXPECTED_PAIRS,test_used=False,
        limitation='selection on previously explored dev clients and fixed grouped data; not final independent test')
    dump(FOLDER/'decision.json',choice)
    text='''DECISÃO ENTRE AS DUAS CANDIDATAS DE DISTILAÇÃO

Escolha pelo critério congelado antes do treino: '''+winner+'\nProfessores: '+str(CANDIDATES[winner])+'''
Aluno global: CNN1D, treinado no servidor e redistribuído aos clientes.

Critério: minimizar a maior perda de AP face à melhor candidata em cada origem.
As diferenças são calculadas dentro da mesma origem, sem equiparar prevalências
dos dois datasets. Desempate: perda média, tempo total e nome. O critério não é
um teste de significância nem permite afirmar superioridade populacional.

AP GLOBAL EM DEV — média das cinco seeds pareadas:
'''+mean.to_string()+'''\n\nPONTUAÇÕES DA DECISÃO:
'''+ranking.to_string()+'''\n\nDIFERENÇAS PAREADAS A − B POR ORIGEM:
'''+differences.to_string(index=False)+'''

São cinco seeds novas, 2031–2035, e os mesmos dados ampliados para ambas as
candidatas, com fit/seleção/holdout agrupados. Dev já foi explorado antes;
não é um conjunto final intocado. O resultado escolhe a configuração para
este trabalho; não demonstra prontidão clínica nem eficácia em novos pacientes.
Não se escolhe a melhor seed nem o checkpoint pela avaliação dev: a escolha
é uma configuração e usa todas as cinco seeds, com a última ronda fixa.
O test continua reservado. Ajustar limiares/hiperparâmetros em validação
interna própria, congelar as decisões e só então avaliar o test final.
'''
    (FOLDER/'decisao_final.txt').write_text(text)
    return choice


def validate(data,require_complete=True):
    validator=isolate('_final_candidates_validation','validation.py')
    validator.SOFT=BASE;validator.signature=signature;validator.soft_domains=lambda _:data
    result=validator.validate_soft(require_complete=False)
    frame,timings=collect()
    if require_complete:assert len(timings)==EXPECTED_PAIRS
    result.update(status='complete' if len(timings)==EXPECTED_PAIRS else 'partial',expected_pairs=EXPECTED_PAIRS,validated_pairs=len(timings),signature=signature())
    dump(BASE/'validation_report.json',result)
    return result


def render():
    import matplotlib.pyplot as plt
    from IPython.display import display,Markdown
    display(Markdown('# Experiência final — candidatas A e B\n\n'
        '**Aluno global CNN1D em ambas.** A: professores AE+CNN / MobileNetV2; B: CNN1D / MobileNetV2 (Icentia / PhysioNet). '
        'Cinco novas seeds pareadas (2031–2035), dez épocas locais por ronda, amostra ampliada da exp4, dez execuções. '
        'Critério pré-fixado: menor perda máxima de AP face ao melhor candidato de cada origem, em dev. '
        'Dev já explorado; test reservado. Esta é a escolha da configuração, não um test final independente.'))
    status=BASE/'status.json';display(json.loads(status.read_text()) if status.exists() else {'status':'not_started'})
    if not (BASE/'summary.json').exists():return
    summary=pd.read_json(BASE/'summary.json');completed=len(collect()[1])
    display(Markdown(f'**{completed}/{EXPECTED_PAIRS} execuções concluídas.**'))
    for split in ['dev','train']:
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        for ax,ds in zip(axes,DOMAINS):
            part=summary[(summary.dataset==ds)&(summary.split==split)&(summary.approach=='global_student')].set_index('candidate')
            values=part.reindex(list(CANDIDATES));x=np.arange(2)
            ax.bar(x,values.pr_auc_ap,yerr=values.pr_auc_std.fillna(0),color=['#2867ad','#c47b27'],capsize=5)
            ax.set_xticks(x,['A: AE+CNN / MobileNet','B: CNN1D / MobileNet']);ax.set_ylim(0,1);ax.set_title(ds+' — '+split)
            ax.set_ylabel('PR-AUC global; média ± desvio entre seeds');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            for i,v in enumerate(values.pr_auc_ap):
                if np.isfinite(v):ax.text(i,min(v+float(values.pr_auc_std.fillna(0).iloc[i])+.02,.96),f'{v:.4f}',ha='center')
        display(fig)
        plots=BASE/'plots';plots.mkdir(parents=True,exist_ok=True);fig.savefig(plots/f'comparison_{split}.png',dpi=140,bbox_inches='tight');plt.close(fig)
    display(summary.round(5))
    decision=FOLDER/'decision.json'
    if decision.exists():
        display(json.loads(decision.read_text()))
        display(pd.read_json(BASE/'paired_differences.json').round(5))
        display(Markdown((FOLDER/'decisao_final.txt').read_text()))
    else:display(Markdown('A decisão final só é produzida após as dez execuções e a validação.'))


def refresh_notebook():
    import base64
    import nbformat
    import IPython.display
    outputs=[]
    def display(value,**kwargs):
        if hasattr(value,'savefig'):
            buffer=io.BytesIO();value.savefig(buffer,format='png',dpi=110,bbox_inches='tight');data={'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':'Comparação A/B'}
        elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False),'text/plain':repr(value)}
        else:data={'text/plain':repr(value)}
        outputs.append(nbformat.v4.new_output('display_data',data=data))
    original=IPython.display.display;IPython.display.display=display
    try:render()
    finally:IPython.display.display=original
    setup='''from pathlib import Path
import sys
PROJECT_ROOT=Path.cwd().resolve()
while not (PROJECT_ROOT/'requirements.txt').exists() and PROJECT_ROOT!=PROJECT_ROOT.parent:
    PROJECT_ROOT=PROJECT_ROOT.parent
sys.path.insert(0,str(PROJECT_ROOT))
'''
    cell=nbformat.v4.new_code_cell('from support.final_candidates import render\nrender()',execution_count=1,outputs=outputs)
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Final candidates — seleção final por distilação'),nbformat.v4.new_code_cell(setup),cell],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.validate(n);nbformat.write(n,FOLDER/'resultados.ipynb')


def run():
    freeze();BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'run.lock').open('a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('A experiência final já está em execução.')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        engine=training_engine()
        try:
            data=prepare()
            for cfg in configurations():
                dump(BASE/'status.json',dict(status='running',candidate=cfg['name'],seed=cfg['seed'],signature=signature()))
                engine.run_pair(cfg,data)
                summarize();refresh_notebook()
            validate(data);choice=summarize(final=True)
            dump(BASE/'status.json',dict(status='complete',validation_passed=True,completed_pairs=EXPECTED_PAIRS,winner=choice['winner'],signature=signature()))
            refresh_notebook()
        except Exception as error:
            dump(BASE/'status.json',dict(status='failed',error=str(error),signature=signature()));refresh_notebook();raise
