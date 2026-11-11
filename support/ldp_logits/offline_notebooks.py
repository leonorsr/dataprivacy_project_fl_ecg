"""Standalone LDP-logit notebooks: no Torch, Flower or closed-loop imports."""
import base64,io,json,os
from pathlib import Path
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/ldp-logits-mpl')
import nbformat
import numpy as np
import pandas as pd
from .experiencias_logits import FONTE_PADRAO,SAIDA_PADRAO,carregar_rondas

ROOT=Path(__file__).resolve().parents[2]
FOLDER=ROOT/'Privacy Techniques/LDP'
TAG='offline_ldp_study'

def image(name):
    from IPython.display import display
    p=SAIDA_PADRAO/'graficos'/name
    if p.exists():display({'image/png':base64.b64encode(p.read_bytes()).decode()},raw=True)

def render_experiments():
    from IPython.display import display,Markdown
    rounds=carregar_rondas(FONTE_PADRAO)
    display(Markdown('## LDP aplicada apenas aos logits — sem estrutura FL\n\n'
        'Este estudo lê mensagens já guardadas e aplica clipping e ruído retrospectivamente. '
        '**Não treina modelos locais/globais, não redistribui modelos e não altera o treino FL em execução.** '
        'A referência é a configuração A/seed 2031, três ficheiros de rondas históricas, 22 clientes × 32 âncoras × duas classes.'))
    first=rounds[0]
    display(pd.DataFrame([dict(dataset=ds,clientes=sum(str(u).startswith(ds+':') for u in first['clientes']),
        peso=float(sum(w for u,w in zip(first['clientes'],first['pesos']) if str(u).startswith(ds+':')))) for ds in ['icentia11k','physionet2017']]))
    display(Markdown('### Desenho\n\n504 configurações: Laplace/Gaussiano × logits originais/centrados/margem binária '
        '× seis epsilons totais (0,5; 1; 2; 4; 8; 16) × 14 limites C. São 200 sorteios por configuração e ronda. '
        'Cada cliente é recortado como um vetor conjunto: L1 para Laplace e L2 para Gaussiano, sensibilidade de substituição 2C. '
        'Os orçamentos são divididos pelas três mensagens; delta total Gaussiano 10⁻⁵, Laplace zero. '
        'Usa-se aleatoriedade reproduzível **apenas para simulação**, não para uma implantação de privacidade.'))
    frame=pd.read_csv(SAIDA_PADRAO/'resumo.csv')
    best=frame.sort_values('consensus_mae_mean').groupby('epsilon_total').first().reset_index()
    display(best[['epsilon_total','mechanism','representation','clip_C','delta_total','consensus_mae_mean','decision_flips_mean','clip_fraction_mean']].round(6))
    for name in ['consensus_mae_vs_epsilon.png','mudancas_decisao_vs_epsilon.png','fracao_clipping_vs_epsilon.png']:image(name)
    display(Markdown('### Métricas deste estudo\n\n'
        'MAE/MSE do consenso e das probabilidades, alterações de decisões nas âncoras e fração afetada pelo clipping. '
        '**Estas discordâncias não são recall, FAR ou erros clínicos**, pois as âncoras não têm rótulos clínicos. '
        'O efeito no PR-AUC de modelos treinados pertence à pasta irmã **LDP + soft labels**, onde o treino continua. '
        'Aplicar ruído às rondas congeladas não recria o feedback que haveria num treino privado.'))

def render_comparison():
    from IPython.display import display,Markdown
    frame=pd.read_csv(SAIDA_PADRAO/'resumo.csv');top=pd.read_csv(SAIDA_PADRAO/'top3_por_epsilon.csv')
    display(Markdown('## Comparação isolada privacidade–fidelidade dos logits\n\n'
        '**Peso atual: 72,73% Icentia11k e 27,27% PhysioNet.** '
        'Menor epsilon representa proteção mais exigente do mecanismo e, em geral, maior ruído. '
        'Não se deve escolher o maior epsilon apenas por apresentar menor erro. Laplace e Gaussiano têm delta diferente.'))
    display(top[['epsilon_total','mechanism','representation','clip_C','delta_total','consensus_mae_mean','consensus_mse_mean','decision_flips_mean']].round(6))
    image('consensus_mae_vs_epsilon.png');image('efeito_de_C.png')
    display(Markdown('### Opções com epsilon total = 4\n\n'
        'Estas três opções foram selecionadas exploratoriamente pelo MAE nos mesmos logits; '
        'não constituem prova de superioridade nem escolha protegida por DP.'))
    selected=top[top.epsilon_total==4]
    display(selected[['mechanism','representation','clip_C','delta_total','consensus_mae_mean','consensus_mae_se_mean','decision_flips_mean']].round(6))
    clipping=pd.read_csv(SAIDA_PADRAO/'somente_clipping.csv');rows=[]
    for r in selected.itertuples():
        q=clipping[(clipping.mechanism==r.mechanism)&(clipping.representation==r.representation)&(clipping.clip_C==r.clip_C)]
        rows.append(dict(mechanism=r.mechanism,representation=r.representation,clip_C=r.clip_C,
            MAE_so_clipping=q.consensus_mae.mean(),MAE_clipping_e_ruido=r.consensus_mae_mean,fracao_clipping=q.clip_fraction.mean()))
    display(pd.DataFrame(rows).round(6))
    display(Markdown('O controlo só com clipping não é privado. Não se somam diretamente os erros de clipping e ruído depois da softmax. '
        'Os quantis e erros-padrão descrevem a aleatoriedade de simulação nos mesmos logits, não generalização clínica.\n\n'
        '**Âmbito:** a proteção estudada é da mensagem conjunta preparada de um cliente sob parâmetros e transformações fixos. '
        'Normalização histórica, participação, seleção de C, métricas e divulgação dos logits originais exigem análise separada. '
        'Este estudo não protege retrospectivamente dados já publicados e não é uma execução FL nem uma divulgação DP de todo o sistema.'))

def render_summary():
    from IPython.display import display,Markdown
    import matplotlib.pyplot as plt
    frame=pd.read_csv(SAIDA_PADRAO/'resumo.csv')
    # Select C within each mechanism/representation/epsilon using the full study.
    keys=['mechanism','representation','epsilon_total']
    selected=frame.loc[frame.groupby(keys)['consensus_mae_mean'].idxmin()].sort_values(keys)
    fig,axes=plt.subplots(1,2,figsize=(12.5,5.7))
    colors={'original':'#2166ac','centrado':'#d97706','margem':'#16876b'}
    names={'original':'Originais','centrado':'Centrados','margem':'Margem'}
    for (mechanism,representation),group in selected.groupby(['mechanism','representation']):
        label=('Laplace' if mechanism=='laplace' else 'Gaussiano')+' · '+names[representation]
        style=dict(color=colors[representation],linestyle='-' if mechanism=='laplace' else '--',
                   marker='o' if mechanism=='laplace' else 's',markersize=4,label=label)
        axes[0].plot(group.epsilon_total,group.consensus_mae_mean,**style)
        axes[1].plot(group.epsilon_total,100*(1-group.decision_flips_mean/32),**style)
    axes[0].set_ylabel('MAE do consenso (menor é melhor)')
    axes[1].set_ylabel('Decisões preservadas (%) — maior é melhor')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([.5,1,2,4,8,16],labels=['0,5','1','2','4','8','16'])
        ax.set_xlabel('Orçamento ε total por cliente, nas três mensagens')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('LDP isolada: compromisso entre privacidade e utilidade',fontsize=15,y=.99)
    axes[0].set_title('Fidelidade das probabilidades')
    axes[1].set_title('Fidelidade da classificação nas 32 âncoras')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,bbox_to_anchor=(.5,.065),frameon=False,fontsize=9)
    fig.text(.5,.025,'← ε menor: maior proteção     |     ε maior: menor proteção →',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.17,1,.94])
    folder=SAIDA_PADRAO/'graficos';folder.mkdir(parents=True,exist_ok=True)
    for extension in ['png','pdf']:
        fig.savefig(folder/('resumo_privacidade_utilidade.'+extension),dpi=200,bbox_inches='tight')
    plt.close(fig)
    display(Markdown('### Síntese do estudo completo — 504 configurações\n\n'
        'Uma figura reúne as duas medidas de utilidade, com os pesos 72,73%/27,27%. '
        'Cada ponto usa o C com menor MAE para esse mecanismo, representação e ε '
        '(200 sorteios por mensagem, três mensagens históricas). '
        '**C pode mudar entre pontos**: são resultados exploratórios selecionados nos mesmos logits, '
        'não uma comparação com clipping fixo.'))
    image('resumo_privacidade_utilidade.png')
    display(Markdown('**Como ler:** à esquerda, menor erro é melhor; à direita, maior preservação é melhor. '
        'As decisões são comparadas com o consenso original, não com rótulos clínicos. '
        'As escalas verticais mostram a variação observada e não começam necessariamente em zero. '
        'Sem proteção, a referência seria MAE=0 e preservação=100%. '
        'Laplace usa δ=0 e Gaussiano δ total=10⁻⁵; o mesmo ε não iguala as duas garantias. '
        'Não estão incluídos resultados de treino FL, SA ou PR-AUC clínico.'))


def write(name,function):
    import IPython.display
    output=[]
    sections=[]
    def capture(value,**kwargs):
        if kwargs.get('raw') and isinstance(value,dict):data=value
        elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False),'text/plain':repr(value)}
        else:data={'text/plain':repr(value)}
        output.append(nbformat.v4.new_output('display_data',data=data))
    old=IPython.display.display;IPython.display.display=capture
    try:
        for title, render in (function if isinstance(function,list) else [(None,function)]):
            start=len(output);render()
            sections.append((title,render,output[start:]))
    finally:IPython.display.display=old
    setup="from pathlib import Path\nimport sys\nroot=Path.cwd().resolve()\nwhile not (root/'requirements.txt').exists() and root!=root.parent: root=root.parent\nsys.path.insert(0,str(root))"
    cells=[nbformat.v4.new_markdown_cell('# LDP nos logits — privacidade e utilidade'),nbformat.v4.new_code_cell(setup)]
    for title,render,outputs in sections:
        if title:cells.append(nbformat.v4.new_markdown_cell('## '+title))
        module='support.ldp_logits.smoke' if render.__name__=='render_smoke' else 'support.ldp_logits.offline_notebooks'
        code='from '+module+' import '+render.__name__+'\n'+render.__name__+'()'
        cells.append(nbformat.v4.new_code_cell(code,execution_count=1,outputs=outputs))
    n=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},TAG:True,'ldp_unified':True})
    nbformat.validate(n);FOLDER.mkdir(parents=True,exist_ok=True);path=FOLDER/name;temp=path.with_suffix('.tmp.ipynb');nbformat.write(n,temp);temp.replace(path)

def refresh():
    from .smoke import render_smoke
    write('experiencias_logits.ipynb',[('1. Gráfico de síntese',render_summary),
        ('2. Dados e estudo completo',render_experiments),
        ('3. Verificação curta e curvas com perfis fixos',render_smoke),
        ('4. Comparação e interpretação',render_comparison)])
    for name in ('comparation_ldp.ipynb','smoke_tests_ldp.ipynb'):
        (FOLDER/name).unlink(missing_ok=True)

if __name__=='__main__':refresh()
