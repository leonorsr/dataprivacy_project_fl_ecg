"""Notebook-only presentation in Privacy Techniques/LDP; raw outputs stay in artifacts."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/ldp-logits-mpl')
import base64,io,json
from pathlib import Path
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from .closed_loop import BASE,FOLDER,conditions,design,complete,folder_for
from .experiencias_logits import SAIDA_PADRAO,FONTE_PADRAO

def show_image(path):
    from IPython.display import display
    if Path(path).exists():display({'image/png':base64.b64encode(Path(path).read_bytes()).decode()},raw=True)

def render_offline():
    from IPython.display import display,Markdown
    display(Markdown('## LDP nos logits congelados — ponderação atual\n\n'
        'Candidata A, seed 2031, três rondas, 22 clientes, 32 âncoras e duas classes. '
        '**Pesos de fit: 72,73% Icentia11k / 27,27% PhysioNet.** '
        'Este primeiro estudo mede a alteração das mensagens, sem treinar modelos. '
        'Os resultados antigos de 50/50 foram preservados no histórico e não são apresentados como atuais.'))
    with np.load(FONTE_PADRAO/'teacher_round_1.npz') as d:
        display(pd.DataFrame([dict(dataset=ds,clientes=sum(str(u).startswith(ds+':') for u in d['client_ids']),
            peso=float(sum(w for u,w in zip(d['client_ids'],d['weights']) if str(u).startswith(ds+':')))) for ds in ['icentia11k','physionet2017']]))
    frame=pd.read_csv(SAIDA_PADRAO/'resumo.csv');top=pd.read_csv(SAIDA_PADRAO/'top3_por_epsilon.csv')
    display(Markdown('### Configuração\n\n504 configurações: Laplace/Gaussiano × original/centrado/margem × seis epsilons totais '
        '(0,5; 1; 2; 4; 8; 16) × 14 limites C, com 200 sorteios por ronda. Clipping conjunto por cliente: '
        'L1 para Laplace e L2 para Gaussiano; sensibilidade de substituição 2C. Epsilon total dividido por três rondas; '
        'delta total Gaussiano 10⁻⁵, Laplace delta zero. Calibração gaussiana analítica. '
        'O mesmo C em espaços ou normas diferentes não significa o mesmo clipping.'))
    best=frame.sort_values('consensus_mae_mean').groupby('epsilon_total').first().reset_index()
    display(best[['epsilon_total','mechanism','representation','clip_C','consensus_mae_mean','decision_flips_mean','clip_fraction_mean']].round(6))
    display(Markdown('### Três opções selecionadas para treino com ε total = 4\n\n'
        'Seleção exploratória pelo MAE do consenso nos mesmos logits, sem consultar labels de avaliação. '
        'Não é uma seleção protegida por DP nem validação independente.'))
    display(top[top.epsilon_total==4][['mechanism','representation','clip_C','delta_total','consensus_mae_mean','decision_flips_mean']].round(6))
    for name in ['consensus_mae_vs_epsilon.png','mudancas_decisao_vs_epsilon.png','efeito_de_C.png','fracao_clipping_vs_epsilon.png']:
        show_image(SAIDA_PADRAO/'graficos'/name)
    display(Markdown('MAE/MSE medem concordância com o consenso anterior; mudanças de decisão nas âncoras não são falsos positivos/negativos clínicos. '
        '**PR-AUC, recall e FAR após LDP são avaliados no segundo notebook, com treino em ciclo fechado e labels reais dos holdouts.** '
        'O RNG deste estudo offline é reproduzível e de simulação; não se deve usá-lo para uma divulgação privada real.'))

def collect():
    rows=[];round_rows=[];counts=[]
    for condition in conditions():
        for seed in design()['seeds']:
            if not complete(condition,seed):continue
            folder=folder_for(condition,seed);f=pd.read_json(folder/'metrics.json');f=f[f.split=='holdout']
            for (round_,role),g in f.groupby(['round','role']):
                r=dict(condition=condition['name'],mode=condition['mode'],seed=seed,round=round_,role=role)
                for metric in ['pr_auc_ap','recall','far','f1','roc_auc']:
                    ok=np.isfinite(g[metric]);r[metric]=float(np.average(g.loc[ok,metric],weights=g.loc[ok,'fit_examples'])) if ok.any() else np.nan
                round_rows.append(r)
                if round_==design()['rounds']:rows.append(r)
            counts.append(dict(condition=condition['name'],seed=seed,path=str(folder),validated=True))
    per=pd.DataFrame(rows);rounds=pd.DataFrame(round_rows);out=BASE/'results';out.mkdir(parents=True,exist_ok=True)
    per.to_json(out/'per_seed.json',orient='records',double_precision=15)
    rounds.to_json(out/'by_round.json',orient='records',double_precision=15)
    (out/'validated_sources.json').write_text(json.dumps(counts,indent=2)+'\n')
    return per,rounds,counts

def metric_figure(per,metric):
    fig,axes=plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
    cs=conditions();names=[c['name'] for c in cs]
    for ax,role in zip(axes,['local','global']):
        for i,c in enumerate(cs):
            q=per[(per.condition==c['name'])&(per.role==role)][metric]
            if q.empty:continue
            error=float(q.std()) if len(q)>1 else 0.
            color={'raw':'#337eaa','clipped':'#929ba7','ldp':'#c58c35'}[c['mode']]
            ax.bar(i,q.mean(),yerr=error,capsize=4,color=color)
            ax.text(i,q.mean()+error+.015,f'{q.mean():.4f}\nn={len(q)}',ha='center',fontsize=8)
        ax.set_xticks(range(len(names)),[n.replace('_','\n') for n in names]);ax.set_ylim(0,1.12)
        ax.set_title('Modelos locais — principal' if role=='local' else 'Modelo global CNN1D');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_ylabel({'pr_auc_ap':'PR-AUC — maior é melhor','recall':'Recall — maior é melhor','far':'FAR — menor é melhor'}[metric])
    fig.supxlabel('Holdout do próprio cliente · média ponderada pelo fit · média ± desvio entre seeds completas',fontsize=9)
    fig.savefig(BASE/'results'/f'{metric}.png',dpi=150,bbox_inches='tight');return fig

def render_training():
    from IPython.display import display,Markdown
    display(Markdown('## Treino com LDP no ciclo cliente–servidor\n\n'
        '**Candidata A:** AE+CNN no Icentia11k, MobileNetV2 no PhysioNet, CNN1D global. Mesmos dados, '
        'cinco seeds 2031–2035, três rondas, dez épocas locais/ronda e três passos globais/ronda. '
        'As 32 âncoras sintéticas são mantidas. Pesos reais de fit 72,73%/27,27%. Sem SA nesta experiência.'))
    cfg=design();display(pd.DataFrame(cfg['profiles']))
    display(Markdown('### Condições\n\nUma referência sem LDP, três controlos só com clipping e três condições com ruído: '
        '**7 condições × 5 seeds = 35 avaliações**. As cinco referências compatíveis já concluídas são revalidadas e reutilizadas; '
        'os outros 30 treinos são novos. Os controlos com clipping **não são privados**. '
        'Nas condições LDP, ε=4 por ciclo de cliente; ε=4/3 por ronda. Gaussianos: δ total=10⁻⁵; Laplace: δ=0. '
        'As garantias dos dois mecanismos não são idênticas apenas por terem o mesmo epsilon.'))
    status=BASE/'status.json';display(json.loads(status.read_text()) if status.exists() else {'status':'not_started'})
    per,rounds,done=collect();display(Markdown(f'**{len(done)}/35 avaliações completas e validadas.** Resultados parciais são identificados por `n`; não se escolhe o melhor checkpoint.'))
    for c in conditions():
        p=BASE/'runs'/c['name']/'status.json'
        if p.exists():display(dict(ldp_condition=c['name'],**json.loads(p.read_text())))
    if not per.empty:
        for metric in ['pr_auc_ap','recall','far']:
            fig=metric_figure(per,metric);display(fig);plt.close(fig)
        display(per.groupby(['condition','role'])[['pr_auc_ap','recall','far','f1','roc_auc']].agg(['mean','std','count']).round(5))
        display(Markdown('Recall e FAR referem-se à classe anomalia com decisão argmax, sem ajuste ao holdout. '
            'AP é a média ponderada dos AP por cliente; não a AP de uma curva de previsões agrupadas. '
            'Os resultados dos modelos locais são dos próprios holdouts. O modelo global é avaliado nos mesmos holdouts. Dev é secundário; test reservado.'))

def render_comparison():
    from IPython.display import display,Markdown
    display(Markdown('## Comparação utilidade–privacidade\n\n'
        'PR-AUC local é principal; global, recall e FAR são complementares. '
        'A comparação **clipping vs sem LDP** mede a alteração por recorte. '
        'A comparação **LDP vs clipping do mesmo perfil** mede o efeito adicional do ruído e do seu feedback nas rondas seguintes.'))
    per,_,done=collect()
    if not per.empty:
        for metric in ['pr_auc_ap','recall','far']:
            fig=metric_figure(per,metric);display(fig);plt.close(fig)
        records=[]
        for metric in ['pr_auc_ap','recall','far']:
            pivot=per.pivot(index=['seed','role'],columns='condition',values=metric)
            for profile in design()['profiles']:
                a=profile['id']+'_clipped';b=profile['id']+'_ldp'
                if a not in pivot or b not in pivot:continue
                paired=pivot[[a,b]].dropna();delta=paired[b]-paired[a]
                for role in ['local','global']:
                    values=delta[delta.index.get_level_values('role')==role]
                    if len(values):records.append(dict(profile=profile['id'],role=role,metric=metric,paired_seeds=len(values),LDP_minus_clipping=values.mean(),std_delta=values.std()))
        if records:display(pd.DataFrame(records).round(6))
    display(Markdown('### Como interpretar a proteção\n\n'
        'O clipping é conjunto sobre toda a mensagem do cliente e a sensibilidade de substituição é 2C. '
        'O ruído do treino usa mecanismos da diffprivlib com aleatoriedade segura, independente das seeds de treino. '
        'Só a mensagem já perturbada e convertida em probabilidades entra na lista de contribuições para agregação; '
        'esta continua a ser uma simulação num processo, sem isolamento real dos clientes. '
        'O registo de orçamento cobra cada divulgação antes da randomização, incluindo tentativas interrompidas. '
        '**ε=4 é por uma execução completa de um cliente, não pela grelha inteira.** '
        'As 15 execuções privadas planeadas por cliente somam conservadoramente ε=60 se todas forem divulgadas; '
        'delta também se compõe nos perfis Gaussianos. A presença de referências e controlos não privados impede apresentar a experiência inteira como uma divulgação ε=4-LDP.\n\n'
        '**Limite do âmbito:** os dados e transformações preparados são fixados. A normalização histórica usa estatísticas agregadas do fit; '
        'não foi demonstrada privacidade end-to-end para a substituição dos dados ECG brutos de um cliente. '
        'A seleção de C, métricas de avaliação, checkpoints e dados originais já divulgados não são automaticamente protegidos. '
        'Os notebooks são resultados de investigação; não são um pacote de resultados DP para publicação de dados clínicos confidenciais.'))
    if len(done)<35:display(Markdown('**Aguardar as cinco seeds de todas as condições antes da conclusão final.**'))
    else:display(Markdown('Todas as condições foram concluídas. Comparar primeiro as diferenças pareadas e a variabilidade; '
        'selecionar pelo desempenho local e compromisso de privacidade, sem tratar a escolha exploratória como avaliação final independente.'))

def render_privacy_utility_summary():
    from IPython.display import display,Markdown
    per,_,done=collect()
    if per.empty:return
    names=['baseline_raw','P1_clipped','P2_clipped','P3_clipped','P1_ldp','P2_ldp','P3_ldp']
    labels=['Sem LDP','P1\nsó clipping','P2\nsó clipping','P3\nsó clipping',
            'P1 · Gaussiano\nε=4; δ=10⁻⁵','P2 · Gaussiano\nε=4; δ=10⁻⁵',
            'P3 · Laplace\nε=4; δ=0']
    display(Markdown('## Síntese: compromisso entre proteção e utilidade\n\n'
        'PR-AUC no holdout do próprio cliente, ponderado pelo fit, na terceira ronda fixa. '
        'Pontos: média; barras: desvio-padrão entre seeds. '
        'São categorias de proteção, **não uma escala numérica de privacidade**. '
        'Todos os perfis privados usam ε=4 por ciclo de cliente.'))
    fig,axes=plt.subplots(1,2,figsize=(15,6))
    for ax,role in zip(axes,['local','global']):
        baseline=per[(per.condition=='baseline_raw')&(per.role==role)].pr_auc_ap.mean()
        ax.axhspan(0,1,xmin=0,xmax=4/7,color='#eef1f4',alpha=.4)
        ax.axvspan(3.5,6.5,color='#e6f3ee',alpha=.7)
        ax.axhline(baseline,color='#555555',linestyle='--',linewidth=1,label='Média sem LDP')
        lows=[];highs=[]
        for i,name in enumerate(names):
            values=per[(per.condition==name)&(per.role==role)].pr_auc_ap
            if values.empty:continue
            mean=values.mean();sd=values.std() if len(values)>1 else 0.
            color='#287f62' if name.endswith('_ldp') else '#316a9a' if name=='baseline_raw' else '#838994'
            ax.errorbar(i,mean,yerr=sd,fmt='o',color=color,capsize=5,markersize=7)
            ax.annotate(f'{mean:.4f}',(i,mean+sd),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
            lows.append(mean-sd);highs.append(mean+sd)
        ax.set_ylim(max(0,min(lows)-.025),min(1,max(highs)+.04))
        ax.set_xlim(-.5,6.5);ax.set_xticks(range(7),labels,fontsize=8)
        ax.set_ylabel('PR-AUC (AP) — maior é melhor')
        ax.set_title('Modelos locais — critério principal' if role=='local' else 'Modelo global CNN1D — complementar')
        ax.grid(axis='y',alpha=.2);ax.legend(loc='lower left',fontsize=8)
    fig.suptitle('LDP + soft labels: utilidade observada e garantia da mensagem',fontsize=15)
    fig.text(.5,.015,'Sem LDP e só clipping: sem garantia LDP  |  Verde: mensagens com LDP, garantias diferentes para δ=0 e δ>0',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.06,1,.94])
    for ext in ['png','pdf']:
        fig.savefig(BASE/'results'/('resumo_privacidade_utilidade.'+ext),dpi=170,bbox_inches='tight')
    display(fig);plt.close(fig)
    local=per[per.role=='local'].groupby('condition').pr_auc_ap.mean()
    rows=[]
    for name in ['P1_ldp','P2_ldp','P3_ldp']:
        if name in local:
            rows.append(dict(perfil=name,PR_AUC_local=local[name],
                diferenca_face_sem_LDP=local[name]-local.get('baseline_raw',np.nan),
                diferenca_face_so_clipping=local[name]-local.get(name.replace('_ldp','_clipped'),np.nan)))
    display(pd.DataFrame(rows).round(6))
    display(Markdown('**Leitura:** compare cada perfil privado com o respetivo controlo de clipping e com a referência sem LDP. '
        'A proximidade do PR-AUC indica preservação de utilidade nesta avaliação; não prova equivalência estatística. '
        'As escalas verticais dos dois painéis diferem para tornar a variação visível. '
        'P1: logits originais/C=0,5; P2: margem/C=0,5; P3: margem/C=2. '
        'Não se pode concluir qual ε oferece o melhor compromisso, pois só ε=4 foi treinado. '
        'ε=4 refere-se a um ciclo, não à grelha inteira; a garantia é da mensagem preparada, '
        'não de todo o processamento dos ECG nem das métricas divulgadas.'))
    render_summary_reading()


def render_summary_reading():
    from IPython.display import display,Markdown
    per,_,_=collect()
    means=per.groupby(['condition','role'])[['pr_auc_ap','recall','far']].mean()
    def value(condition,metric='pr_auc_ap',role='local'):
        return float(means.loc[(condition,role),metric])
    display(Markdown(
        '### Leitura do gráfico e escolha segundo a aplicação\n\n'
        f"**Utilidade local.** A referência sem LDP apresenta PR-AUC={value('baseline_raw'):.4f}. "
        f"Com proteção, P2 obtém {value('P2_ldp'):.4f}, P1 {value('P1_ldp'):.4f} "
        f"e P3 {value('P3_ldp'):.4f}. Face à referência, as diferenças são "
        f"{value('P2_ldp')-value('baseline_raw'):+.4f}, "
        f"{value('P1_ldp')-value('baseline_raw'):+.4f} e "
        f"{value('P3_ldp')-value('baseline_raw'):+.4f}, respetivamente. "
        'São diferenças absolutas de PR-AUC, não percentagens. P1 e P2 preservam melhor a média '
        'local do que P3 nesta experiência. As barras representam desvio entre seeds, não intervalos '
        'de confiança; o gráfico não demonstra superioridade estatística nem equivalência.\n\n'
        '**Efeito adicional do ruído.** Comparando com o controlo de clipping do mesmo perfil, '
        f"o PR-AUC local muda {value('P1_ldp')-value('P1_clipped'):+.4f} em P1, "
        f"{value('P2_ldp')-value('P2_clipped'):+.4f} em P2 e "
        f"{value('P3_ldp')-value('P3_clipped'):+.4f} em P3. "
        'A pequena melhoria média de P2 não prova que o ruído melhora o modelo. '
        'O resultado de P3 indica maior perda média local neste perfil específico; '
        'não permite concluir que Laplace é sempre pior, pois também mudam a norma e C.\n\n'
        f"**Modelo global.** P3 tem a maior média entre os perfis privados "
        f"({value('P3_ldp',role='global'):.4f}), próxima da referência "
        f"({value('baseline_raw',role='global'):.4f}). "
        'Isto não compensa automaticamente a perda local: neste projeto, o critério principal '
        'é o PR-AUC dos modelos dos próprios clientes.\n\n'
        '**Se a aplicação privilegia deteção local em wearables:** P2 é a candidata exploratória '
        'mais favorável pelo PR-AUC local médio entre os perfis privados. '
        f"Apresenta também recall local={value('P2_ldp','recall'):.4f}, "
        f"com FAR={value('P2_ldp','far'):.4f}. "
        'O gráfico não mede autonomia, memória ou latência no wearable; esses custos exigem testes próprios.\n\n'
        '**Se a prioridade é reduzir falsos alarmes:** P1 merece comparação com P2, '
        f"pois apresenta FAR local={value('P1_ldp','far'):.4f}, contra "
        f"{value('P2_ldp','far'):.4f} de P2, com recall "
        f"{value('P1_ldp','recall'):.4f} contra {value('P2_ldp','recall'):.4f}. "
        'Estes valores vêm das tabelas de métricas e usam a decisão fixa argmax/0,5. '
        'A escolha clínica exige definir o custo de falhar anomalias e o de gerar alarmes, '
        'e validar o limiar num conjunto próprio.\n\n'
        '**Se o requisito de privacidade exige δ=0:** P3 é o único dos três perfis privados '
        'testados que satisfaz esse requisito, com Laplace e ε=4, aceitando a menor média local observada. '
        'P1 e P2 usam ε=4 e δ=10⁻⁵. Este requisito mais estrito em δ não estabelece '
        'uma ordenação universal de segurança entre as implementações. '
        'Sem LDP e só clipping são referências de utilidade, não alternativas que satisfaçam um requisito LDP.\n\n'
        '**Orientação para este projeto:** levar P2 como candidata principal quando se aceita '
        '(ε=4, δ=10⁻⁵) e se privilegia PR-AUC local; manter P1 como alternativa se a redução '
        'de falsos alarmes for prioritária; considerar P3 se δ=0 for obrigatório. '
        'A decisão é exploratória nas mesmas cinco seeds: falta validação independente e '
        'o gráfico não permite escolher outro ε. Uma pequena perda de PR-AUC não demonstra '
        'resistência a ataques. As garantias referem-se às mensagens preparadas e ao orçamento '
        'por ciclo, sem proteger automaticamente métricas, checkpoints ou todo o processamento dos ECG.'))


def outputs_for(function):
    import IPython.display
    outputs=[]
    def capture(value,**kwargs):
        if kwargs.get('raw') and isinstance(value,dict):data=value
        elif hasattr(value,'savefig'):
            b=io.BytesIO();value.savefig(b,format='png',dpi=125,bbox_inches='tight');data={'image/png':base64.b64encode(b.getvalue()).decode(),'text/plain':'LDP: utilidade e privacidade'}
        elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        elif hasattr(value,'to_html'):data={'text/html':value.to_html(),'text/plain':repr(value)}
        else:data={'text/plain':repr(value)}
        outputs.append(nbformat.v4.new_output('display_data',data=data))
    old=IPython.display.display;IPython.display.display=capture
    try:function()
    finally:IPython.display.display=old
    return outputs

def refresh():
    folder=FOLDER.with_name('LDP + soft labels');folder.mkdir(parents=True,exist_ok=True)
    setup="from pathlib import Path\nimport sys\nroot=Path.cwd().resolve()\nwhile not (root/'requirements.txt').exists() and root!=root.parent: root=root.parent\nsys.path.insert(0,str(root))"
    for name,function in [('experiencias_treino_ldp.ipynb',render_training),('comparation_ldp.ipynb',render_comparison)]:
        code='from support.ldp_logits.notebooks import '+function.__name__+'\n'+function.__name__+'()'
        n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# '+name.removesuffix('.ipynb').replace('_',' ')),
            nbformat.v4.new_code_cell(setup),nbformat.v4.new_code_cell(code,execution_count=1,outputs=outputs_for(function))],
            metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'ldp_design':design()})
        if name=='comparation_ldp.ipynb':
            n.cells.extend([nbformat.v4.new_markdown_cell('## Gráfico final de privacidade–utilidade'),
                nbformat.v4.new_code_cell('from support.ldp_logits.notebooks import render_privacy_utility_summary\nrender_privacy_utility_summary()',
                    execution_count=2,outputs=outputs_for(render_privacy_utility_summary))])
        nbformat.validate(n);path=folder/name;temp=path.with_suffix('.tmp.ipynb');nbformat.write(n,temp);temp.replace(path)
    from .offline_notebooks import refresh as refresh_offline
    refresh_offline()

if __name__=='__main__':refresh()
