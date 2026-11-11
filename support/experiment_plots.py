"""Comparative charts only: read validated summaries, never change training signatures."""
import base64
import io
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/icentia-mpl')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat

ROOT=Path(__file__).resolve().parents[1]
BASES={'fedavg':ROOT/'FedAvg tests/artifacts/global_model_v1',
       'soft':ROOT/'SoftLabels tests/artifacts/global_student_v1'}
NAMES={'Autoencoder_CNN_GradCAM':'AE+CNN (Grad-CAM)','CNN1D':'CNN1D',
       'CNN1D_Optimized':'CNN1D otimizada','CNN_BiLSTM':'CNN+BiLSTM',
       'CNN_BiLSTM_Attention':'CNN+BiLSTM+Attention','CNN_BiLSTM_CBAM':'CNN+BiLSTM+CBAM',
       'MobileNetV2':'MobileNetV2','ResNet34':'ResNet34','LogReg':'Regressão logística','SVM':'SVM linear'}
STRATEGIES={'common':'Modelo comum','per_dataset':'Modelo por dataset','per_client':'Modelo por cliente'}
TAG='global_comparison_chart'

def label(value):
    if value in STRATEGIES:return STRATEGIES[value]
    if value.startswith('IC_') and '__PH_' in value:
        a,b=value[3:].split('__PH_');return 'IC: '+NAMES.get(a,a)+' | PH: '+NAMES.get(b,b)
    if '__' in value:
        source,model=value.split('__',1)
        if source in {'icentia11k','physionet2017'}:return NAMES.get(model,model)
    return NAMES.get(value,value)

def figure(kind,split,experiment=None):
    assert kind in BASES and split in {'train','dev'}
    source=BASES[kind]/'summary.json';summary=pd.read_json(source)
    part=summary[summary.split==split]
    if kind=='soft':part=part[part.experiment==experiment]
    if part.empty:return None
    key='model' if kind=='fedavg' else 'configuration'
    primary=('local_after_fedavg_training' if split=='train' else 'local_federated_external') if kind=='fedavg' else 'local_teacher'
    if kind=='fedavg':
        phases=[('local_after_fedavg_training' if split=='train' else 'local_federated_external','Local antes da agregação — principal','#c47b27'),
                ('global','Global redistribuído','#2867ad'),
                ('local_only' if split=='train' else 'local_only_external','Treino privado','#7b858f')]
    else:
        phases=[('local_teacher','Professor local — principal','#c47b27'),
                ('global_student','Aluno global CNN1D','#2867ad'),
                ('local_control','Controlo privado','#7b858f')]
    max_rows=max(int(((part.dataset==ds)&(part.approach==primary)).sum()) for ds in ['icentia11k','physionet2017'])
    if max_rows==0:return None
    fig,axes=plt.subplots(1,2,figsize=(24,max(6.5,.49*max_rows+2.4)),layout='constrained')
    for ax,ds in zip(axes,['icentia11k','physionet2017']):
        subset=part[part.dataset==ds]
        ranking=subset[subset.approach==primary].sort_values(['pr_auc_ap','roc_auc',key],ascending=[False,False,True])
        order=ranking[key].tolist();positions=np.arange(len(order))
        assert len(order)==len(set(order))
        for j,(approach,title,color) in enumerate(phases):
            values=subset[subset.approach==approach].set_index(key).reindex(order)
            if values.empty:continue
            means=values.pr_auc_ap.to_numpy(float);spread=values.pr_auc_std.fillna(0).to_numpy(float)
            ax.barh(positions+(j-1)*.25,means,height=.23,xerr=spread,color=color,
                    label=title,capsize=2,error_kw={'elinewidth':.8,'alpha':.8})
            for y,value,error in zip(positions+(j-1)*.25,means,spread):
                if np.isfinite(value):ax.text(min(value+float(error)+.012,.98),y,f'{value:.3f}',va='center',fontsize=8 if j==0 else 7,fontweight='bold' if j==0 else 'normal')
        ax.set_yticks(positions,[label(v) for v in order],fontsize=8 if max_rows>15 else 9)
        ax.invert_yaxis();ax.set_xlim(0,1.05)
        ax.set_title(('Icentia11k' if ds=='icentia11k' else 'PhysioNet 2017'),fontsize=13)
        ax.set_xlabel('PR-AUC da anomalia (average precision)')
        ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        ax.legend(loc='lower right',fontsize=8)
    seeds=sorted(set(part.seeds.dropna().astype(int)))
    spread_text='uma seed; sem estimativa de dispersão' if seeds==[1] else 'média ± desvio amostral entre '+ '/'.join(map(str,seeds))+' seeds'
    title=('FedAvg' if kind=='fedavg' else str(experiment).upper())+' — '+('holdouts locais' if split=='train' else 'dev: clientes fora do treino')
    fig.suptitle(title+'\n'+spread_text+' · ordenação pelo modelo local em cada dataset',fontsize=14)
    destination=BASES[kind]/'plots'/f'comparison_{experiment or "fedavg"}_{split}.png'
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=130,bbox_inches='tight')
    return fig,destination

def render_comparison(kind,split='dev',experiment=None):
    from IPython.display import display,Markdown
    result=figure(kind,split,experiment)
    if result is None:
        display(Markdown('Ainda não existem resultados para este gráfico.'));return
    fig,_=result
    try:display(fig)
    finally:plt.close(fig)

def refresh_charts():
    for kind,name in [('fedavg','FedAvg tests/comparation_fedavg.ipynb'),('soft','SoftLabels tests/comparation_softlabels.ipynb')]:
        path=ROOT/name;notebook=nbformat.read(path,4)
        notebook.cells=[c for c in notebook.cells if TAG not in c.get('metadata',{}).get('tags',[])]
        source=BASES[kind]/'summary.json'
        if not source.exists():continue
        summary=pd.read_json(source)
        experiments=[None] if kind=='fedavg' else [e for e in ['exp1','exp2','exp3','exp4','exp5'] if e in set(summary.experiment)]
        caption=('## Gráficos comparativos\n\nBarras separadas para global, local e referência privada. '
            'As posições são ordenadas pela PR-AUC local dentro de cada dataset, sem misturar as duas origens. '
            'As barras de erro representam desvio entre seeds, não intervalos de confiança populacionais. ')
        if kind=='soft':
            caption+=('IC e PH indicam as arquiteturas dos professores Icentia e PhysioNet; o aluno global permanece CNN1D. '
                'Na exp4, os nomes representam regras de seleção dos professores.')
        else:
            caption+=('Local refere-se ao modelo após treino da última ronda, antes da agregação; '
                'global é o modelo agregado e redistribuído, da mesma arquitetura indicada na linha.')
        cells=[nbformat.v4.new_markdown_cell(caption,metadata={'tags':[TAG]})]
        count=0
        for experiment in experiments:
            for split in ['dev','train']:
                result=figure(kind,split,experiment)
                if result is None:continue
                fig,destination=result
                try:
                    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=110,bbox_inches='tight')
                    data=base64.b64encode(buffer.getvalue()).decode()
                finally:plt.close(fig)
                title=('FedAvg' if kind=='fedavg' else experiment.upper())+' — '+('dev' if split=='dev' else 'holdout local')
                cells.append(nbformat.v4.new_markdown_cell('### '+title,metadata={'tags':[TAG]}))
                code='from support.experiment_plots import render_comparison\n'+f'render_comparison({kind!r}, split={split!r}, experiment={experiment!r})'
                cells.append(nbformat.v4.new_code_cell(code,execution_count=1,
                    outputs=[nbformat.v4.new_output('display_data',data={'image/png':data,'text/plain':title})],metadata={'tags':[TAG]}))
                count+=1
        # Show charts immediately after the existing root/package bootstrap, ahead of tables.
        notebook.cells[2:2]=cells
        nbformat.validate(notebook);nbformat.write(notebook,path)
        print('GRÁFICOS GUARDADOS',path,':',count,flush=True)
