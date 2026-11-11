"""Plot measured utility costs alongside the experiment's privacy mechanisms."""
import base64
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/softlabels-comparison-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat
from support.softlabels_privacy_comparison import OUT, ROOT, SOURCES


def main():
    frame = pd.read_csv(OUT / 'por_seed.csv')
    labels = ['Sem SA / sem LDP', 'Só SA', 'Só LDP (P2)', 'SA + LDP (P2)']
    colors = ['#49677f', '#179d91', '#d79531', '#9269ad']
    fig = plt.figure(figsize=(14, 9))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, .75, .53], hspace=.55, wspace=.25)
    records = []
    for col, role in enumerate(['local', 'global']):
        ax = fig.add_subplot(grid[0, col])
        base = frame[(frame.role == role) & (frame.condition == labels[0])].set_index('seed').pr_auc_ap
        for i, condition in enumerate(labels):
            values = frame[(frame.role == role) & (frame.condition == condition)].set_index('seed').pr_auc_ap
            mean, std = values.mean(), values.std(ddof=1)
            loss = (base - values) * 100
            records.append(dict(condition=condition, role=role, ap_mean=mean, ap_std=std,
                                loss_ap_points_mean=loss.mean(), loss_ap_points_std=loss.std(ddof=1),
                                utility_retained_percent=100 * mean / base.mean()))
            ax.errorbar(i, mean, yerr=std, fmt='o', color=colors[i], markersize=10, capsize=6, lw=2)
            ax.text(i, 1.035, f'{mean:.4f}', ha='center', fontsize=11, color=colors[i], weight='bold')
        ax.axhline(base.mean(), color=colors[0], linestyle='--', alpha=.5, label='Média sem SA / sem LDP')
        ax.set_title('Utilidade local' if role == 'local' else 'Utilidade global', weight='bold', pad=30)
        ax.set_ylabel('PR-AUC (AP) ↑ · média ± desvio padrão')
        ax.set_xticks(range(4), ['Sem SA\nsem LDP', 'Só SA', 'Só LDP', 'SA + LDP'])
        ax.set_ylim(0, 1)
        ax.set_xlim(-.5, 3.5)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower left', fontsize=9, frameon=False)
    result = pd.DataFrame(records)
    result.to_csv(OUT / 'tradeoff_utilidade_privacidade.csv', index=False)
    ax = fig.add_subplot(grid[1, :]); ax.axis('off')
    rows = [
        ['Oculta contribuições individuais\nno protocolo de agregação (SA)', 'Não', 'Sim¹', 'Não', 'Sim¹'],
        ['Perturbação local com garantia DP\nna mensagem preparada (LDP)', 'Não', 'Não', 'Sim²', 'Sim²'],
        ['Perda média de AP local\nface à referência (pontos ×100)', *[f'{result[(result.role == "local") & (result.condition == c)].iloc[0].loss_ap_points_mean:+.3f}' for c in labels]],
        ['Perda média de AP global\nface à referência (pontos ×100)', *[f'{result[(result.role == "global") & (result.condition == c)].iloc[0].loss_ap_points_mean:+.3f}' for c in labels]],
    ]
    table = ax.table(cellText=rows, colLabels=['Proteção / custo de utilidade', *labels],
                     colWidths=[.36, .16, .16, .16, .16], cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d8e0e7')
        if r == 0:
            cell.set_facecolor('#edf2f6'); cell.set_text_props(weight='bold')
        elif c == 0:
            cell.set_text_props(ha='left'); cell.set_facecolor('#f7f9fb')
        elif r <= 2:
            cell.set_facecolor('#e2f2ed' if rows[r-1][c].startswith('Sim') else '#f2f3f5')
    ax = fig.add_subplot(grid[2, :]); ax.axis('off')
    ax.text(0, 1, 'Leitura do compromisso', fontsize=12, weight='bold', va='top')
    ax.text(0, .77,
            'SA + LDP combina as duas proteções com uma perda média de AP local de 0,297 pontos face à referência.\n'
            'As diferenças são pequenas face à dispersão entre seeds; não demonstram superioridade ou equivalência estatística.\n'
            '¹ SA: simulação com servidor semi-honesto, 24 bits e limiar 12/22; não fornece um orçamento ε.\n'
            '² LDP P2: ε=4, δ=10⁻⁵ por cliente/ciclo; cinco seeds compõem ε=20, δ=5×10⁻⁵ por condição privada.\n'
            'SA não reduz ε nem o ruído LDP. O ruído independente impede isolar o efeito da SA entre condições privadas.\n'
            'Garantia da mensagem preparada, não do ECG bruto de ponta a ponta. Holdout, ronda 3, cinco seeds; não test.',
            fontsize=10, va='top', linespacing=1.6)
    fig.suptitle('Soft labels — trade-off entre utilidade e privacidade', fontsize=19, weight='bold', y=.98)
    fig.text(.5, .935, 'Quatro configurações categóricas: SA e LDP são proteções distintas, sem escala numérica comum de privacidade', ha='center', fontsize=11, color='#425466')
    fig.subplots_adjust(top=.82, bottom=.045, left=.07, right=.98)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'tradeoff_utilidade_privacidade.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def update_notebook():
    path = ROOT / 'Privacy Techniques/LDP + SA/experiencias_ldp_sa.ipynb'
    nb = nbformat.read(path, as_version=4)
    marker = 'softlabels_privacy_tradeoff'
    nb.cells = [c for c in nb.cells if c.metadata.get('comparison_section') != marker]
    intro = nbformat.v4.new_markdown_cell('''## Trade-off entre utilidade e privacidade

O gráfico relaciona a **utilidade medida (PR-AUC)** com os **mecanismos de proteção** de cada configuração. SA e LDP não são níveis de uma mesma escala: SA oculta contribuições durante a agregação, nas hipóteses do protocolo; LDP limita a informação da mensagem preparada através de perturbação local. A combinação mantém o mesmo orçamento LDP e acrescenta SA.

A tabela apresenta a perda de AP face à referência, multiplicada por 100 (valores positivos indicam perda). As barras de erro são desvios padrão entre cinco seeds, não intervalos de confiança. Usam-se os mesmos holdouts da comparação anterior; não se misturam métricas de test.
''')
    code = nbformat.v4.new_code_cell('''from support.softlabels_privacy_tradeoff import main as criar_tradeoff
from support.softlabels_privacy_comparison import OUT
from IPython.display import display, Image

# Regenera apenas a figura a partir dos resultados guardados, sem treino.
criar_tradeoff()
display(Image(filename=str(OUT / "tradeoff_utilidade_privacidade.png")))
''')
    code.outputs = [nbformat.v4.new_output('display_data', data={'image/png': base64.b64encode((OUT / 'tradeoff_utilidade_privacidade.png').read_bytes()).decode()})]
    conclusion = nbformat.v4.new_markdown_cell('''**Interpretação:** SA + LDP combina as duas proteções com AP local média de **0,7989**, face a **0,8018** sem proteção adicional: perda observada de **0,297 pontos de AP ×100** (cerca de **99,63% da AP de referência**). A AP global passa de **0,4410** para **0,4335**, perda de **0,754 pontos**. A dispersão entre seeds é maior do que estas diferenças; não se conclui equivalência estatística nem que SA melhora a utilidade.

**Âmbito da privacidade:** ε=4 e δ=10⁻⁵ são por cliente/ciclo. As cinco seeds privadas compõem ε=20 e δ=5×10⁻⁵ por condição, além de divulgações e tentativas anteriores. Executar e divulgar ambas as condições privadas exige compor também os respetivos orçamentos. SA não reduz esse orçamento. A garantia LDP é da mensagem preparada, sem demonstração de privacidade de ponta a ponta do ECG bruto. A SA é uma simulação com servidor semi-honesto e não estabelece resistência a servidor ativamente malicioso ou a envenenamento.

Assim, **SA + LDP é um compromisso plausível para combinar as duas proteções preservando a utilidade local observada**, condicionado às hipóteses acima; este estudo não quantifica uma pontuação global de privacidade nem demonstra uma solução ótima.
''')
    for cell in [intro, code, conclusion]:
        cell.metadata['comparison_section'] = marker
        nb.cells.append(cell)
    nbformat.validate(nb)
    nbformat.write(nb, path)


if __name__ == '__main__':
    main()
    update_notebook()
