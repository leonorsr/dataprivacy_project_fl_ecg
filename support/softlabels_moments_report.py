"""Append the moment-compression study to comparation_softlabels without erasing earlier cells."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/soft-moments-mpl')
import base64,io,json
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from support.softlabels_moments import BASE,MODES,DESIGN,complete,folder_for,signature,ROOT

TAG='softlabels_moments_comparison'
LABELS={'probabilities64':'Atual\n64 probabilidades','anomaly32':'Determinístico\n32 probabilidades','mc_mean32':'MC Dropout\n32 médias','mc_meanstd64':'MC Dropout\n32 médias + 32 std'}
EXPLANATION="""## Nova experiência — média e desvio-padrão por âncora com MC Dropout

A proposta escolhida envia uma média e um desvio-padrão **por âncora**, calculados sobre 20 previsões estocásticas da **probabilidade de anomalia da mesma âncora**. A média continua associada à classe anomalia; `p(normal) = 1 − média`. O desvio-padrão mede variabilidade das previsões, não é calculado entre as duas classes nem entre âncoras diferentes. A opção anterior de dois números por cliente foi interrompida e preservada em `artifacts/softlabels_moments_history/client_wide_v1/`; não é misturada nesta experiência.

| Representação por cliente/ronda | Números float32 | Payload sem contagem de fit | Objetivo |
|---|---:|---:|---|
| Atual: normal e anomalia por âncora | 64 | 256 B | Referência determinística |
| Só anomalia por âncora | 32 | 128 B | Eliminar redundância binária |
| MC Dropout: só média de anomalia por âncora | 32 | 128 B | Isolar o efeito de repetir previsões |
| MC Dropout: média e std de anomalia por âncora | 64 | 256 B | Usar também a variabilidade na aprendizagem |

**Exemplo:** previsões de anomalia `[0,80; 0,85; 0,90]` na mesma âncora dão média 0,85 e std aproximadamente 0,041. O servidor associa essa âncora a `[normal=0,15; anomalia=0,85]`, mantendo a informação de classe. Isto não garante a mesma decisão de uma única previsão determinística; usa a previsão média. Também não se preserva toda a distribuição com apenas dois momentos.

**MC Dropout:** o professor faz 20 passagens com as camadas Dropout ativas. As restantes camadas, incluindo BatchNorm, permanecem em avaliação; as âncoras não atualizam as suas estatísticas. O estado do gerador aleatório é restaurado após as passagens, e as duas variantes MC usam as mesmas sementes de amostragem. O aumento de cálculo é contabilizado. [Fundamento de MC Dropout — Gal e Ghahramani, 2016](https://proceedings.mlr.press/v48/gal16.html).

**Agregação por âncora:** `mu = sum(w_i*mu_i)` e `var = sum(w_i*(sigma_i²+mu_i²)) − mu²`. Esta variância da mistura inclui a variabilidade dentro de cada professor e o desacordo entre professores. Não é a variância estatística da média do ensemble, nem uma medida de confiança calibrada.

**Uso do std no aluno:** a média fornece o alvo suave `[1−mu, mu]`. Na variante média+std, a entropia cruzada de cada âncora é ponderada por `1/(1+4*var)`, normalizada para média 1 entre as 32 âncoras. A constante 4 vem do limite de variância 1/4 de uma probabilidade em [0,1]; a regra é fixada antes de avaliar os resultados, sem ajustar ao holdout. A variante só com médias MC usa a mesma entropia cruzada sem essa ponderação, permitindo separar o efeito das previsões repetidas do efeito do std.

**Desenho e critérios:** candidata A (AE+CNN no Icentia e MobileNetV2 no PhysioNet), CNN1D global, mesmos dados agrupados, cinco seeds 2031–2035, três rondas e dez épocas locais. Pesos de fit 72,73%/27,27%. PR-AUC local no holdout do próprio cliente é principal; PR-AUC global nos mesmos holdouts é secundário. Usa-se a última ronda fixa, médias ponderadas e diferenças pareadas. Dev é secundário, test reservado e holdout exploratório. As referências concluídas só são reutilizadas quando compatíveis e novamente validadas.

Esta ablação é **sem SA/LDP**, para isolar a representação. Para uma futura SA, os clientes poderiam preparar momentos aditivos (`mu` e `sigma²+mu²`) em vez de somar desvios-padrão diretamente. Média+std por âncora mantém os mesmos 64 valores da referência, logo não se presume poupança de bytes; a vantagem potencial é informação sobre variabilidade. Nenhuma variante demonstra maior privacidade ou menor consumo energético apenas pelo formato da mensagem. O download da CNN1D global é mantido e contabilizado nas quatro variantes.
"""

def tables():
    per=[];cost=[];statuses=[]
    for mode in MODES:
        for seed in DESIGN['seeds']:
            if not complete(mode,seed):continue
            folder=folder_for(mode,seed);meta=json.loads((folder/'run.json').read_text())
            f=pd.read_json(folder/'metrics.json');f=f[(f['round']==DESIGN['rounds'])&(f.split=='holdout')]
            for role,g in f.groupby('role'):
                ok=np.isfinite(g.pr_auc_ap)
                per.append(dict(mode=mode,seed=seed,role=role,pr_auc_ap=float(np.average(g.loc[ok,'pr_auc_ap'],weights=g.loc[ok,'fit_examples'])),
                    valid_clients=int(ok.sum()),total_clients=len(g),fit_coverage=float(g.loc[ok,'fit_examples'].sum()/g.fit_examples.sum())))
            c=pd.read_json(folder/'communication.json')
            cost.append(dict(mode=mode,seed=seed,upload_bytes=int(c.upload_bytes.sum()),download_bytes=int(c.download_bytes.sum()),
                total_bytes=int(c.total_bytes.sum()),local_compute_seconds=float(c.local_compute_seconds.sum())))
            statuses.append(dict(mode=mode,seed=seed,path=str(folder.relative_to(ROOT)),signature=meta['signature'],validated=True))
    per=pd.DataFrame(per);cost=pd.DataFrame(cost);paired=pd.DataFrame()
    if not per.empty:
        pivot=per.pivot(index=['seed','role'],columns='mode',values='pr_auc_ap')
        if 'probabilities64' in pivot:
            rows=[]
            for mode in ['anomaly32','mc_mean32','mc_meanstd64']:
                if mode in pivot:
                    q=pivot[['probabilities64',mode]].dropna().reset_index()
                    for r in q.to_dict('records'):rows.append(dict(mode=mode,seed=r['seed'],role=r['role'],delta_vs_current=r[mode]-r['probabilities64']))
            paired=pd.DataFrame(rows)
    result=BASE/'results';result.mkdir(parents=True,exist_ok=True)
    for f,name in [(per,'per_seed'),(cost,'communication_per_seed'),(paired,'paired_differences')]:f.to_csv(result/f'{name}.csv',index=False)
    (result/'validated_sources.json').write_text(json.dumps(statuses,indent=2)+'\n')
    return per,cost,paired,statuses

def figure(per,role):
    f=per[per.role==role];fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for i,mode in enumerate(MODES):
        v=f[f['mode']==mode].pr_auc_ap
        if not len(v):continue
        std=float(v.std()) if len(v)>1 else 0.
        ax.bar(i,v.mean(),yerr=std,capsize=5,color=['#337eaa','#49a17d','#9670b5','#cc8b3d'][i])
        ax.text(i,v.mean()+std+.02,f'{v.mean():.4f}\nn={len(v)}',ha='center')
    ax.set_xticks(range(len(MODES)),[LABELS[m] for m in MODES]);ax.set_ylim(0,1.1);ax.set_ylabel('PR-AUC ponderado pelo fit')
    ax.set_title('Professores locais — principal' if role=='local' else 'Aluno global CNN1D');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.supxlabel('Holdout do próprio cliente · média ± desvio entre seeds completas',fontsize=9)
    return fig

def render():
    from IPython.display import display,Markdown
    per,cost,paired,statuses=tables()
    display(Markdown(EXPLANATION));state=BASE/'status.json'
    display(json.loads(state.read_text()) if state.exists() else {'status':'not_started'})
    display(Markdown(f'**{len(statuses)}/20 condições completas e validadas.** `n` mostra as seeds realmente disponíveis; resultados incompletos não permitem uma decisão final.'))
    for mode in ['anomaly32','mc_mean32','mc_meanstd64']:
        p=BASE/mode/'status.json'
        if p.exists():display(dict(representation=mode,**json.loads(p.read_text())))
    if not per.empty:
        for role in ['local','global']:
            fig=figure(per,role);display(fig);fig.savefig(BASE/'results'/f'pr_auc_{role}.png',dpi=150,bbox_inches='tight');plt.close(fig)
        display(per)
    if not cost.empty:
        summary=cost.groupby('mode')[['upload_bytes','download_bytes','total_bytes']].mean()/(22*1024)
        summary=summary.reindex(MODES);display(Markdown('### Comunicação média por cliente no ciclo de três rondas — KiB'));display(summary)
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');x=np.arange(len(MODES))
        ax.bar(x,summary.upload_bytes,label='Upload');ax.bar(x,summary.download_bytes,bottom=summary.upload_bytes,label='Download')
        ax.set_xticks(x,[LABELS[m] for m in MODES]);ax.set_ylabel('KiB/cliente/ciclo');ax.legend();ax.set_title('Inclui distribuição inicial e três modelos globais')
        display(fig);fig.savefig(BASE/'results/communication.png',dpi=150,bbox_inches='tight');plt.close(fig)
    if not paired.empty:display(paired)
    if len(statuses)==20:
        mean=per.groupby(['mode','role']).pr_auc_ap.mean().unstack('role')
        display(Markdown('### Comparação concluída\n\nAs quatro representações foram avaliadas nas mesmas cinco seeds. '
            'A leitura principal é o PR-AUC local e a diferença pareada; o PR-AUC global e os bytes mostram o compromisso. '
            'A variante MC média+std mantém a identidade das âncoras e usa o mesmo payload de 64 valores da referência. '
            'A variante de 32 valores reduz o payload em 50% com equivalência matemática binária; não se deve atribuir significado estatístico a pequenas diferenças numéricas sem análise adicional.'))
        display(mean)

def refresh():
    import IPython.display
    outputs=[]
    def capture(value,**kwargs):
        if hasattr(value,'savefig'):
            b=io.BytesIO();value.savefig(b,format='png',dpi=130,bbox_inches='tight')
            data={'image/png':base64.b64encode(b.getvalue()).decode(),'text/plain':'Comparação das representações de soft labels'}
        elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False),'text/plain':repr(value)}
        else:data={'text/plain':repr(value)}
        outputs.append(nbformat.v4.new_output('display_data',data=data))
    original=IPython.display.display;IPython.display.display=capture
    try:render()
    finally:IPython.display.display=original
    bootstrap="from pathlib import Path\nimport sys\nroot=Path.cwd().resolve()\nwhile not (root/'requirements.txt').exists() and root!=root.parent: root=root.parent\nsys.path.insert(0,str(root))\n"
    cell=nbformat.v4.new_code_cell(bootstrap+'from support.softlabels_moments_report import render\nrender()',execution_count=1,outputs=outputs,metadata={'tags':[TAG]})
    title=nbformat.v4.new_markdown_cell('## Comparação das mensagens: probabilidades vs média e std por âncora (MC Dropout)',metadata={'tags':[TAG]})
    n=nbformat.v4.new_notebook(cells=[title,cell],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.validate(n);nbformat.write(n,BASE/'experiencias_moments.ipynb')
    from support.softlabels_overview import refresh as refresh_overview
    refresh_overview()

if __name__=='__main__':refresh()
