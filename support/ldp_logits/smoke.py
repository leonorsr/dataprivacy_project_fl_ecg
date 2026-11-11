"""Exploratory privacy–utility sweep on frozen, fit-weighted logits only."""
import base64
import io
import numpy as np
import pandas as pd
from .experiencias_logits import carregar_rondas, FONTE_PADRAO, estatisticas
from .ldp_logits import (softmax_binario, representar, para_logits,
                         recortar_por_cliente, parametros_ruido)

EPSILONS = [.5, 1., 2., 4., 8., 16.]
PROFILES = [('gaussiano','original',.5), ('gaussiano','margem',.5), ('laplace','margem',2.)]

def render_smoke():
    from IPython.display import display, Markdown
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rounds = carregar_rondas(FONTE_PADRAO)
    rng = np.random.default_rng(2031)
    display(Markdown('## Smoke test: compromisso privacidade–utilidade nos logits\n\n'
        'Estudo exploratório sem treino nem arquitetura FL: três mensagens históricas, '
        'três perfis fixos, seis valores de ε total e 20 sorteios por perfil, ε e mensagem. '
        'Os pesos são verificados: 16/22 Icentia (72,73%) e 6/22 PhysioNet (27,27%).\n\n'
        '**Privacidade:** ε menor corresponde a proteção mais forte, para o mesmo mecanismo e δ. '
        'O orçamento por cliente é dividido pelas três mensagens. Laplace usa δ=0; Gaussiano usa '
        'δ total=10⁻⁵, pelo que não se devem equiparar os mecanismos apenas pelo ε. '
        '**Utilidade:** fidelidade ao consenso original, medida por MAE e percentagem de decisões preservadas; '
        'não é desempenho clínico.'))
    rows, controls = [], []
    for r in rounds:
        weights = r['pesos']
        for dataset, count in [('icentia11k',16), ('physionet2017',6)]:
            mask = np.array([str(i).startswith(dataset+':') for i in r['clientes']])
            assert mask.sum() == count and np.isclose(weights[mask].sum(),count/22)
        for mechanism, representation, limit in PROFILES:
            profile = f'{mechanism} / {representation} / C={limit:g}'
            clipped, flags = recortar_por_cliente(representar(r['logits'],representation),limit,
                                                  1 if mechanism=='laplace' else 2)
            metrics = estatisticas(softmax_binario(para_logits(clipped,representation))[None],
                                   r['prob'],r['consenso'],weights)
            controls.append(dict(perfil=profile,ronda=r['ronda'],
                MAE_consenso=float(metrics['consensus_mae'][0]),
                decisoes_preservadas_pct=100*(1-float(metrics['decision_flips'][0])/32),
                clientes_com_clipping_pct=100*flags.mean()))
            for epsilon in EPSILONS:
                _,_,scale = parametros_ruido(mechanism,limit,epsilon/3,
                                             1e-5/3 if mechanism=='gaussiano' else 0.)
                noise = (rng.laplace if mechanism=='laplace' else rng.normal)(
                    0.,scale,size=(20,)+clipped.shape)
                probs = softmax_binario(para_logits(clipped[None]+noise,representation))
                assert probs.shape==(20,22,32,2) and np.isfinite(probs).all()
                assert np.allclose(probs.sum(axis=-1),1)
                metrics = estatisticas(probs,r['prob'],r['consenso'],weights)
                for draw in range(20):
                    rows.append(dict(perfil=profile,epsilon_total=epsilon,ronda=r['ronda'],sorteio=draw,
                        MAE_consenso=float(metrics['consensus_mae'][draw]),
                        decisoes_preservadas_pct=100*(1-float(metrics['decision_flips'][draw])/32)))
    frame = pd.DataFrame(rows)
    summary = frame.groupby(['perfil','epsilon_total']).agg(
        MAE_consenso=('MAE_consenso','mean'), MAE_desvio_simulacao=('MAE_consenso','std'),
        decisoes_preservadas_pct=('decisoes_preservadas_pct','mean')).reset_index()
    control = pd.DataFrame(controls).groupby('perfil').mean(numeric_only=True).drop(columns='ronda')
    display(Markdown('### Referências para interpretar a perda de utilidade\n\n'
        'Sem ruído e sem clipping: MAE=0 e decisões preservadas=100% por definição. '
        'A tabela seguinte usa **apenas clipping**, sem garantia LDP, para separar a distorção do clipping da do ruído.'))
    display(control.reset_index())
    display(Markdown('### Resultados com LDP simulado\n\n'
        'Médias sobre três mensagens históricas e 20 sorteios por mensagem. '
        'O desvio descreve variabilidade de simulação nestes ficheiros, não incerteza de generalização clínica.'))
    display(summary)
    fig, axes = plt.subplots(1,2,figsize=(13,4.5))
    for profile, group in summary.groupby('perfil'):
        for ax, metric in zip(axes,['MAE_consenso','decisoes_preservadas_pct']):
            line, = ax.plot(group.epsilon_total,group[metric],marker='o',label=profile)
            ax.axhline(control.loc[profile,metric],color=line.get_color(),linestyle=':',alpha=.6)
    for ax, title, baseline in zip(axes,['Erro do consenso (menor é melhor)',
                                         'Decisões preservadas (%)'],[0,100]):
        ax.axhline(baseline,color='black',linestyle='--',alpha=.5)
        ax.set_xscale('log',base=2);ax.set_xticks(EPSILONS,labels=[str(e) for e in EPSILONS])
        ax.set_xlabel('ε total por cliente — à esquerda: maior privacidade')
        ax.set_title(title);ax.grid(alpha=.2)
    axes[1].set_ylim(0,105)
    axes[0].legend(fontsize=7)
    fig.suptitle('LDP nos logits: utilidade vs orçamento de privacidade')
    fig.tight_layout()
    buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140,bbox_inches='tight');plt.close(fig)
    display({'image/png':base64.b64encode(buffer.getvalue()).decode()},raw=True)
    display(Markdown('Linhas pontilhadas: controlo só com clipping; linha preta tracejada: '
        'referência original sem proteção. Compare os valores de ε **dentro de cada perfil**, '
        'mantendo C e representação fixos. Maior ε permite menor ruído, mas clipping pode limitar a recuperação '
        'da utilidade; flutuações não monotónicas são possíveis neste smoke test pequeno.\n\n'
        '**PASSOU:** pesos, consensos e probabilidades validados. Este teste permite observar o compromisso '
        'entre proteção da mensagem e fidelidade ao consenso; não determina sozinho o melhor ε para uso clínico. '
        'PR-AUC, recall e FAR exigem dados com rótulos e avaliação dos modelos. '
        'O RNG reproduzível é apenas de simulação; ε não contabiliza a divulgação de toda a grelha. '
        'Os logits já guardados não ficam protegidos retroativamente.'))

def main():
    from .offline_notebooks import refresh
    refresh()

if __name__=='__main__':
    main()
