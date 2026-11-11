"""Reports distinguish the trained global student from local teachers."""
import json
import pandas as pd
from .shared import *

def collect():
    frames=[]
    for path in SOFT.glob('*/*/*/run_config.json'):
        cfg=json.loads(path.read_text())
        if cfg.get('status')!='complete' or cfg.get('signature')!=signature():continue
        f=pd.read_json(path.with_name('metrics.json'))
        f['experiment']=cfg['experiment'];f['configuration']=cfg['name'];f['seed']=cfg['seed'];frames.append(f)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def build():
    f=collect()
    if f.empty:return
    metrics=['pr_auc_ap','roc_auc','recall','far','f1']
    keys=['experiment','configuration','dataset','split','approach']
    per=f.groupby(keys+['seed'])[metrics].mean().reset_index()
    per.to_json(SOFT/'per_seed.json',orient='records',double_precision=15)
    s=per.groupby(keys)[metrics].mean().reset_index()
    spread=per.groupby(keys).agg(pr_auc_std=('pr_auc_ap','std'),seeds=('seed','nunique')).reset_index()
    s.merge(spread,on=keys).to_json(SOFT/'summary.json',orient='records',double_precision=15)

def render_fedavg():
    from IPython.display import display,Markdown
    display(Markdown('# FedAvg corrigido — modelo global redistribuído\n\n'
        'Apenas modelos com parâmetros compatíveis; RF/KNN e as duas GRU excluídos. '
        'Três rondas, três seeds de treino e dados fixos à seed 2026. '
        'O servidor agrega parâmetros ponderados pelo fit e redistribui o modelo em cada ronda. '
        'As réplicas finais dos clientes têm exatamente o mesmo estado que o global. '
        'Não há ensemble de previsões neste protocolo. Resultados anteriores são históricos.'))
    state=FED/'status.json';display(json.loads(state.read_text()) if state.exists() else {'status':'not_started'})
    path=FED/'summary.json'
    if path.exists():
        f=pd.read_json(path)
        for split in ['train','dev']:
            display(Markdown('## Modelo global entregue aos clientes — '+split))
            display(f[(f.approach=='global')&(f.split==split)].sort_values(['dataset','pr_auc_ap'],ascending=[True,False]).round(5))
    selected=FED/'selection.json'
    if selected.exists():display(json.loads(selected.read_text()))
    display(Markdown('Ranking para selecionar candidatos: AP do modelo global entregue, nos holdouts locais; '
        'ROC-AUC desempata. Os modelos locais pré-agregação são diagnóstico separado. '
        'Holdout já usado na pesquisa: seleção exploratória; test reservado.'))

def render_soft():
    from IPython.display import display,Markdown
    display(Markdown('# Soft labels corrigidas — aluno CNN1D global\n\n'
        '**Professores locais → logits nas âncoras → CNN1D treinada no servidor → pesos globais '
        'redistribuídos → orientação dos professores na ronda seguinte.** '
        'CNN1D com trunk e classificador partilhados e adaptadores de features por origem (17/19 entradas). '
        'Esta adaptação permite um único modelo com entradas nativas diferentes. '
        'O global é agora um checkpoint treinado, nunca a média das previsões de avaliação. '
        'RF/KNN e GRU excluídos. Não confundir com os resultados históricos de consenso sem aluno global.'))
    state=SOFT/'status.json';display(json.loads(state.read_text()) if state.exists() else {'status':'not_started'})
    build();path=SOFT/'summary.json'
    if not path.exists():return
    f=pd.read_json(path)
    for exp in ['exp1','exp2','exp3','exp4','exp5','pilot']:
        part=f[f.experiment==exp]
        if part.empty:continue
        display(Markdown('## '+exp+' — rankings por dataset'))
        for approach in ['global_student','local_teacher','local_control']:
            display(Markdown('### '+approach+' em dev'))
            display(part[(part.approach==approach)&(part.split=='dev')].sort_values(['dataset','pr_auc_ap'],ascending=[True,False]).round(5))
    display(Markdown('O ranking global e o ranking dos professores locais respondem a perguntas diferentes. '
        'A escolha final anterior de RF foi retirada: avaliava outro fluxo. '
        'Dev/holdout são exploratórios; não concluir superioridade sem os novos resultados completos. '
        'Três seeds na grelha e cinco na exp4 medem variabilidade de treino com dados fixos. '
        'Controlos privados não igualam FLOPs/tempo. Test reservado.'))
