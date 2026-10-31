"""Local-first final charts; does not change or restart the frozen training code."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/icentia-mpl')
import base64,fcntl,io,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat
from support import final_candidates as final

TAG='local_priority_comparison'

def render():
    from IPython.display import display,Markdown
    display(Markdown('## PR-AUC dos modelos locais — comparação prioritária\n\n'
        '**Professor local** em destaque; aluno global e controlo privado apresentados separadamente. '
        'Valores numéricos nas três barras. Holdout avalia cada professor nos seus próprios exemplos; '
        'dev avalia os professores nos clientes dev da mesma origem. '
        'As médias são iguais entre clientes por origem/seed e depois entre seeds. '
        '**Critério final: PR-AUC local em dev**, confirmado pelo utilizador. '
        'O global é complementar. Resultados parciais não são uma decisão final.'))
    path=final.BASE/'summary.json'
    if not path.exists():
        display(Markdown('Ainda não há execuções completas para calcular métricas locais.'));return
    summary=pd.read_json(path)
    for split in ['dev','train']:
        fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
        for ax,ds in zip(axes,final.DOMAINS):
            part=summary[(summary.dataset==ds)&(summary.split==split)]
            pos=np.arange(2)
            for j,(approach,label,color) in enumerate([
                ('local_teacher','Local — principal','#c47b27'),
                ('global_student','Aluno global CNN1D','#2867ad'),
                ('local_control','Controlo privado','#7b858f')]):
                values=part[part.approach==approach].set_index('candidate').reindex(list(final.CANDIDATES))
                mean=values.pr_auc_ap;std=values.pr_auc_std.fillna(0)
                ax.bar(pos+(j-1)*.25,mean,width=.23,yerr=std,capsize=3,label=label,color=color)
                for i,(v,error) in enumerate(zip(mean,std)):
                    if np.isfinite(v):ax.text(pos[i]+(j-1)*.25,min(v+float(error)+.015,.96),f'{v:.4f}',ha='center',fontsize=9,fontweight='bold' if j==0 else 'normal')
            ax.set_xticks(pos,['A: AE+CNN / MobileNet','B: CNN1D / MobileNet']);ax.set_ylim(0,1.08)
            ax.set_title(ds+' — '+('holdout do próprio cliente' if split=='train' else 'dev'))
            ax.set_ylabel('PR-AUC; média ± desvio entre seeds');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(fontsize=8,loc='lower right')
        display(fig)
        folder=final.BASE/'plots';folder.mkdir(exist_ok=True);fig.savefig(folder/f'local_priority_{split}.png',dpi=140,bbox_inches='tight');plt.close(fig)
        table=summary[(summary.split==split)&(summary.approach.isin(['local_teacher','global_student','local_control']))]
        display(table[['candidate','dataset','approach','pr_auc_ap','pr_auc_std','recall','far','seeds']].round(5))


def refresh():
    import IPython.display
    outputs=[]
    def display(value,**kwargs):
        if hasattr(value,'savefig'):
            b=io.BytesIO();value.savefig(b,format='png',dpi=110,bbox_inches='tight');data={'image/png':base64.b64encode(b.getvalue()).decode(),'text/plain':'PR-AUC local/global/controlo'}
        elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False),'text/plain':repr(value)}
        else:data={'text/plain':repr(value)}
        outputs.append(nbformat.v4.new_output('display_data',data=data))
    previous=IPython.display.display;IPython.display.display=display
    try:render()
    finally:IPython.display.display=previous
    path=final.FOLDER/'resultados.ipynb';notebook=nbformat.read(path,4)
    notebook.cells=[c for c in notebook.cells if TAG not in c.get('metadata',{}).get('tags',[])]
    notebook.cells.insert(2,nbformat.v4.new_code_cell('from support.local_priority import render\nrender()',execution_count=1,outputs=outputs,metadata={'tags':[TAG]}))
    nbformat.validate(notebook);nbformat.write(notebook,path)


def watch():
    previous=None
    while True:
        path=final.BASE/'summary.json';notebook=final.FOLDER/'resultados.ipynb'
        try:
            state=(path.stat().st_mtime_ns if path.exists() else 0,notebook.stat().st_mtime_ns)
            if state!=previous:
                refresh();previous=(path.stat().st_mtime_ns if path.exists() else 0,notebook.stat().st_mtime_ns)
            lock=(final.BASE/'run.lock').open('a+')
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
                from support.local_decision import finalize
                if finalize():final.refresh_notebook()
                refresh();print('LOCAL PRIORITY: decision and notebook updated; training stopped.',flush=True);return
            except BlockingIOError:pass
            finally:lock.close()
        except (OSError,ValueError) as error:print('Will refresh local charts again:',error,flush=True)
        time.sleep(15)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    watch() if a.watch else refresh()
