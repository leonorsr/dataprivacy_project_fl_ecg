"""Save communication and local/global AP comparisons, including embedded notebook figures."""
import base64,io,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat
from support.sa_communication import BASE,DESIGN,METRICS,signature

LABELS={'Fedavg_plain':'FedAvg','Fedavg_SA':'FedAvg + SA',
        'Softlabels_plain':'Soft Labels','Softlabels_SA':'Soft Labels + SA'}

WEARABLE_JUSTIFICATION="""## Justificação da opção por Soft Labels + SA para wearables

Para o objetivo de utilização em wearables, privilegia-se **distilação federada com agregação segura (Soft Labels + SA)** pela redução da comunicação e pelo suporte a modelos locais heterogéneos. A escolha representa um compromisso entre os recursos do dispositivo e o desempenho preditivo; não resulta de as Soft Labels terem obtido o melhor PR-AUC. Neste desenho, o FedAvg apresentou melhores médias de PR-AUC local e global.

### O que é efetivamente transmitido

Na implementação avaliada, cada cliente envia **duas probabilidades, normal e anomalia, por âncora sintética**. Com 32 âncoras, são **64 valores por ronda**, antes das mensagens adicionais da SA. Não são apenas dois parâmetros de média e desvio-padrão (`mean` e `std`). O FedAvg transmite cerca de 20 mil valores do estado do modelo. Enviar apenas média e desvio-padrão seria outra variante, ainda não implementada nem avaliada nestes resultados.

| Comunicação por cliente no ciclo de três rondas, incluindo SA | FedAvg + SA | Soft Labels + SA |
|---|---:|---:|
| Upload | 253,91 KiB | 15,54 KiB |
| Download | 333,36 KiB | 333,36 KiB |
| Total | 587,27 KiB | 348,90 KiB |

As Soft Labels + SA reduzem aproximadamente **94% do upload** e **41% da comunicação total** relativamente a FedAvg + SA. A redução total é menor porque ambas as abordagens continuam a distribuir a CNN1D global. São bytes dos payloads lógicos do protocolo, sem cabeçalhos de rede, retransmissões ou medições numa rede real. Menos comunicação é relevante para dispositivos com conectividade limitada, mas esta experiência não mediu consumo de bateria.

### Segurança e limites da conclusão

Nas duas abordagens, a SA destina-se a ocultar as mensagens individuais e permitir ao servidor recuperar o agregado, dentro das hipóteses do protocolo. As Soft Labels permitem partilhar conhecimento sem transmitir os parâmetros completos dos professores. Contudo, **menos valores transmitidos não demonstram maior privacidade**: as probabilidades também podem revelar informação. Não foram avaliados ataques de inferência ou reconstrução, nem foi aplicada LDP nesta comparação. Os testes funcionais de uma simulação local não constituem uma validação de segurança para produção.

### Adequação ao dispositivo

A heterogeneidade permite escolher professores adequados a diferentes dispositivos. Isso não garante que os professores atuais, AE+CNN e MobileNetV2, sejam apropriados para execução diretamente num wearable: nesta experiência, o seu cálculo local foi mais demorado do que o da CNN1D usada no FedAvg. A avaliação seguinte deverá medir **memória, energia, tempo de treino e inferência num dispositivo representativo**, juntamente com o PR-AUC local.

**Justificação para o relatório:** Seleciona-se a distilação federada com agregação segura pela redução do volume de comunicação e pelo suporte a modelos locais heterogéneos. Esta escolha representa um compromisso com o desempenho preditivo observado, não uma demonstração de superioridade em privacidade ou consumo energético. A adequação a wearables permanece condicionada à avaliação dos modelos e do protocolo no hardware de destino.
"""

def collect():
    metrics=[];costs=[];runs=[]
    for kind in ['Fedavg','Softlabels']:
        for path in (BASE/kind/'artifacts').glob('*/*/run.json'):
            meta=json.loads(path.read_text())
            if meta.get('signature')!=signature():continue
            validation=path.with_name('validation.json')
            valid=validation.exists() and json.loads(validation.read_text()).get('signature')==meta['signature']
            keys=dict(algorithm=kind,condition=meta['condition'],seed=meta['seed'])
            for name,frames in [('metrics.json',metrics),('communication.json',costs)]:
                p=path.with_name(name)
                if p.exists():
                    f=pd.read_json(p)
                    for key,value in keys.items():f[key]=value
                    f['run_complete']=meta['status']=='complete' and valid
                    frames.append(f)
            if meta['status']=='complete' and valid:
                runs.append(dict(**keys,elapsed_seconds=meta['elapsed_seconds']))
    return (pd.concat(metrics,ignore_index=True) if metrics else pd.DataFrame(),
            pd.concat(costs,ignore_index=True) if costs else pd.DataFrame(),pd.DataFrame(runs))

def tables():
    frame,cost,runs=collect();per=[]
    if not frame.empty:
        last=frame[(frame['round']==DESIGN['primary_round'])&frame.run_complete&(frame.split=='holdout')]
        for keys,g in last.groupby(['algorithm','condition','seed','role']):
            row=dict(zip(['algorithm','condition','seed','role'],keys))
            for metric in METRICS:
                ok=np.isfinite(g[metric]);assert ok.any()
                row[metric]=float(np.average(g.loc[ok,metric],weights=g.loc[ok,'fit_examples']))
            row['valid_ap_clients']=int(np.isfinite(g.pr_auc_ap).sum());row['total_clients']=len(g)
            row['ap_fit_coverage']=float(g.loc[np.isfinite(g.pr_auc_ap),'fit_examples'].sum()/g.fit_examples.sum())
            per.append(row)
    per=pd.DataFrame(per);comm=pd.DataFrame();paired=pd.DataFrame()
    if not cost.empty:
        comm=cost[cost.run_complete].groupby(['algorithm','condition','seed'])[[
            'upload_bytes','download_bytes','total_bytes','aggregation_seconds','local_compute_seconds']].sum().reset_index()
    if not per.empty:
        pivot=per.pivot(index=['algorithm','seed','role'],columns='condition',values='pr_auc_ap')
        if {'SA','plain'}<=set(pivot.columns):
            paired=pivot.dropna(subset=['SA','plain']).reset_index();paired['SA_minus_plain']=paired.SA-paired.plain
    for kind in ['Fedavg','Softlabels']:
        result=BASE/kind/'results';result.mkdir(parents=True,exist_ok=True)
        for f,name in [(frame,'client_metrics'),(cost,'communication_by_round'),(per,'per_seed'),(comm,'communication_per_seed'),(paired,'paired_sa_effect')]:
            selected=f[f.algorithm==kind] if not f.empty else f
            selected.to_csv(result/f'{name}.csv',index=False)
    result=BASE/'results';result.mkdir(exist_ok=True)
    per.to_csv(result/'per_seed.csv',index=False);comm.to_csv(result/'communication_per_seed.csv',index=False)
    paired.to_csv(result/'paired_sa_effect.csv',index=False)
    return frame,cost,runs,per,comm,paired

def bar_figure(per,role,kind=None):
    f=per[per.role==role]
    if kind:f=f[f.algorithm==kind]
    if f.empty:return None
    grouped=f.groupby(['algorithm','condition']).pr_auc_ap.agg(['mean','std','count'])
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for i,((algorithm,condition),row) in enumerate(grouped.iterrows()):
        std=0 if pd.isna(row['std']) else row['std'];color='#2c78a8' if condition=='plain' else '#d58d36'
        ax.bar(i,row['mean'],yerr=std,capsize=5,color=color)
        ax.text(i,row['mean']+std+.02,f"{row['mean']:.4f}\nn={int(row['count'])}",ha='center',fontsize=9)
    ax.set_xticks(range(len(grouped)),[LABELS[a+'_'+c] for a,c in grouped.index]);ax.set_ylim(0,1.1)
    ax.set_ylabel('PR-AUC ponderado pelo fit');ax.set_title('Modelos locais — principal' if role=='local' else 'Modelo global CNN1D')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.supxlabel('Holdout do próprio cliente · última ronda fixa · média ± desvio entre seeds',fontsize=9)
    return fig

def cost_figure(comm,kind=None):
    f=comm if kind is None else comm[comm.algorithm==kind]
    if f.empty:return None
    grouped=f.groupby(['algorithm','condition'])[['upload_bytes','download_bytes']].mean()/(22*1024)
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained');x=np.arange(len(grouped))
    ax.bar(x,grouped.upload_bytes,label='Cliente → servidor')
    ax.bar(x,grouped.download_bytes,bottom=grouped.upload_bytes,label='Servidor → cliente')
    ax.set_xticks(x,[LABELS[a+'_'+c] for a,c in grouped.index]);ax.set_ylabel('KiB por cliente no ciclo de três rondas')
    ax.set_title('Comunicação — mensagens do protocolo e modelos');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    return fig

def render(kind=None):
    from IPython.display import display,Markdown
    frame,cost,runs,per,comm,paired=tables()
    state=BASE/'status.json';status=json.loads(state.read_text()) if state.exists() else {'status':'not_started'}
    display(Markdown('# Comunicação e PR-AUC — '+(kind or 'FedAvg vs Soft Labels')+'\n\n'
        '**Comparação emparelhada com/sem SA dentro de cada abordagem.** Mesmos IDs, cinco seeds (2031–2035), três rondas e dez épocas locais. '
        '22 clientes, pesos de fit: 72,73% Icentia e 27,27% PhysioNet. Sem desistências nesta comparação. '
        'SA: soma única, 24 bits, limiar 12. FedAvg usa CNN1D comum; Soft Labels usa professores AE+CNN/MobileNetV2 e aluno CNN1D. '
        'A diferença entre abordagens inclui arquitetura/objetivo e custo de treino distintos; as épocas iguais não igualam FLOPs.'))
    display(status);display(Markdown(f'**{len(runs)}/20 execuções concluídas e validadas.** Barras parciais só incluem execuções completas; `n` indica seeds disponíveis.'))
    display(Markdown('PR-AUC local: professor no próprio holdout nas Soft Labels; rede local após treino, antes da agregação, em FedAvg. '
        'PR-AUC global: o modelo global avaliado nos mesmos holdouts. Média ponderada dos AP individuais, não uma curva conjunta. '
        'O modelo FedAvg entregue ao cliente após agregação corresponde à barra global. Dev é secundário e está nas tabelas por cliente; test não é usado. '
        'AP indefinida num holdout com uma só classe é excluída, com cobertura explícita nas tabelas.'))
    if not per.empty:
        for role in ['local','global']:
            fig=bar_figure(per,role,kind)
            if fig:
                display(fig);folder=BASE/(kind or '')/'results';fig.savefig(folder/f'pr_auc_{role}.png',dpi=150,bbox_inches='tight');plt.close(fig)
        display(per if kind is None else per[per.algorithm==kind])
    if not comm.empty:
        fig=cost_figure(comm,kind)
        if fig:
            display(fig);fig.savefig(BASE/(kind or '')/'results/communication.png',dpi=150,bbox_inches='tight');plt.close(fig)
        display(comm if kind is None else comm[comm.algorithm==kind])
    if not paired.empty:display(paired if kind is None else paired[paired.algorithm==kind])
    display(Markdown('Comunicação: soma dos payloads de upload e download; inclui chaves, partilhas e desmascaramento da SA, '
        'distribuição inicial e três distribuições globais (uma após cada ronda). Contagem por mensagem lógica, sem cabeçalhos TCP/TLS nem retransmissões. '
        'Âncoras sintéticas geradas localmente; preparação inicial dos dados não contabilizada. Tempos medidos localmente, sem latência de rede. '
        'FedAvg: parâmetros com sinal codificados por transformação afim com limite público ±256, sem clipping silencioso; buffers inteiros são arredondados. '
        'Soft Labels: probabilidades ponderadas em [0,1]. Os 24 bits não representam o mesmo erro absoluto nos dois espaços. '
        'SA não oferece LDP nem esta simulação constitui uma validação de segurança para produção.'))

def refresh():
    import IPython.display
    tables()
    for kind,name in [(None,'comparation_communication.ipynb'),('Fedavg','Fedavg/experiencias_fedavg.ipynb'),('Softlabels','Softlabels/experiencias_softlabels.ipynb')]:
        outputs=[]
        def capture(value,**kwargs):
            if hasattr(value,'savefig'):
                buffer=io.BytesIO();value.savefig(buffer,format='png',dpi=130,bbox_inches='tight')
                data={'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':'Comparação Comunicação/PR-AUC'}
            elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
            elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False),'text/plain':repr(value)}
            else:data={'text/plain':repr(value)}
            outputs.append(nbformat.v4.new_output('display_data',data=data))
        original=IPython.display.display;IPython.display.display=capture
        try:render(kind)
        finally:IPython.display.display=original
        bootstrap="from pathlib import Path\nimport sys\nroot=Path.cwd().resolve()\nwhile not (root/'requirements.txt').exists() and root!=root.parent: root=root.parent\nsys.path.insert(0,str(root))"
        code='from support.sa_communication_report import render\n'+f'render({kind!r})'
        notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Experiências de comunicação com e sem SA'),
            nbformat.v4.new_code_cell(bootstrap),nbformat.v4.new_code_cell(code,execution_count=1,outputs=outputs)],
            metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
        if kind is None:
            notebook.cells.append(nbformat.v4.new_markdown_cell(WEARABLE_JUSTIFICATION,metadata={'tags':['wearable_justification']}))
        if kind=='Softlabels':
            from support.sa_softlabels_security import append_section
            append_section(notebook)
        nbformat.validate(notebook);nbformat.write(notebook,BASE/name)

if __name__=='__main__':refresh()
