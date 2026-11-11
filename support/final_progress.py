"""Read-only intermediate evaluation of atomic final-candidate checkpoints."""
import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR','/tmp/icentia-mpl')
import base64,contextlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import nbformat
from support import final_candidates as final
from support.federated_global import personalized,shared
from support.federated_global.soft import GlobalCNN1D,predictions
from global_fed.metrics import evaluate

TAG='partial_final_progress'

def measure():
    # Read source caches through the grouped loader; never train or modify checkpoints.
    with contextlib.redirect_stdout(io.StringIO()):data=final.prepare()
    rows=[]
    for cfg in final.configurations():
        folder=final.BASE/'confirmation'/cfg['name']/str(cfg['seed'])
        meta=folder/'run_config.json'
        if not meta.exists():continue
        config=json.loads(meta.read_text())
        if config.get('signature')!=final.signature():continue
        if config.get('status')=='complete':continue
        for r in range(1,4):
            checkpoint=folder/f'global_round_{r}.pt'
            if not checkpoint.exists():continue
            model=GlobalCNN1D({ds:len(d['config']['feature_names']) for ds,d in data.items()},cfg['seed'])
            model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
            for ds,d in data.items():
                for c in d['clients'].values():
                    if c['split']!='dev':continue
                    ix=c['hold_indices'];p=predictions(model,'global_student',ds,c,ix)
                    metrics,_=evaluate(c['y'][ix],p,ds)
                    rows.append(dict(candidate=cfg['name'],seed=cfg['seed'],round=r,dataset=ds,client=c['uid'],**metrics))
    frame=pd.DataFrame(rows)
    if frame.empty:return frame
    summary=frame.groupby(['candidate','seed','round','dataset'])[['pr_auc_ap','roc_auc','recall','far','f1']].mean().reset_index()
    directory=final.BASE/'monitor';directory.mkdir(exist_ok=True)
    summary.to_json(directory/'partial_global_dev_metrics.json',orient='records',double_precision=15)
    return summary


def render_partial():
    from IPython.display import display,Markdown
    display(Markdown('## Progresso parcial do aluno global\n\n'
        '**Métricas reais de checkpoints intermédios, apenas na execução em curso.** '
        'Não são a comparação final A/B nem médias das cinco seeds. Não selecionar a melhor ronda por este gráfico; '
        'a decisão final continua a usar a última ronda fixa e todas as seeds.'))
    torch.set_num_threads(2)
    summary=measure()
    if summary.empty:
        display(Markdown('Ainda não existe checkpoint intermédio disponível para avaliar.'));return
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,ds in zip(axes,shared.DOMAINS):
        for (candidate,seed),part in summary[summary.dataset==ds].groupby(['candidate','seed']):
            part=part.sort_values('round');label=('A' if candidate.startswith('A') else 'B')+' — seed '+str(seed)
            ax.plot(part['round'],part.pr_auc_ap,'o-',label=label)
            for r,v in zip(part['round'],part.pr_auc_ap):ax.annotate(f'{v:.4f}',(r,v),xytext=(0,6),textcoords='offset points',ha='center')
        ax.set_xticks([1,2,3]);ax.set_xlim(.75,3.25);ax.set_ylim(0,1)
        ax.set_title(ds+' — dev');ax.set_xlabel('Ronda já concluída');ax.set_ylabel('PR-AUC do aluno global');ax.grid(alpha=.2);ax.legend()
    display(fig)
    fig.savefig(final.BASE/'monitor/partial_progress.png',dpi=140,bbox_inches='tight');plt.close(fig)
    display(summary.round(5))
    display(Markdown('Esta observação é descritiva e não altera treinos, checkpoints, critérios ou hiperparâmetros.'))


def refresh_partial():
    import IPython.display
    outputs=[]
    def display(value,**kwargs):
        if hasattr(value,'savefig'):
            b=io.BytesIO();value.savefig(b,format='png',dpi=110,bbox_inches='tight');data={'image/png':base64.b64encode(b.getvalue()).decode(),'text/plain':'Progresso parcial real'}
        elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False),'text/plain':repr(value)}
        else:data={'text/plain':repr(value)}
        outputs.append(nbformat.v4.new_output('display_data',data=data))
    original=IPython.display.display;IPython.display.display=display
    try:render_partial()
    finally:IPython.display.display=original
    path=final.FOLDER/'resultados.ipynb';n=nbformat.read(path,4)
    n.cells=[c for c in n.cells if TAG not in c.get('metadata',{}).get('tags',[])]
    n.cells.append(nbformat.v4.new_code_cell('from support.final_progress import render_partial\nrender_partial()',execution_count=1,outputs=outputs,metadata={'tags':[TAG]}))
    nbformat.validate(n);nbformat.write(n,path)
