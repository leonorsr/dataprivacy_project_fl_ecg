"""Visualize configured privacy guarantees, not unmeasured attack resistance."""
import base64
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/softlabels-comparison-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nbformat
from support.softlabels_privacy_comparison import ROOT, OUT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = ['Sem SA\nsem LDP', 'Só SA', 'Só LDP\n(P2)', 'SA + LDP\n(P2)']
    colors = ['#49677f', '#179d91', '#d79531', '#9269ad']
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    protections = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    ax = axes[0, 0]
    ax.imshow(protections, cmap=ListedColormap(['#eef0f3', '#58ae98']), vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(4), labels)
    ax.set_yticks([0, 1], ['SA¹', 'LDP²'])
    ax.set_title('A · Proteções presentes no protocolo', weight='bold', pad=15)
    for r in range(2):
        for c in range(4):
            ax.text(c, r, 'Sim' if protections[r,c] else 'Não', ha='center', va='center', weight='bold', fontsize=13)
    ax = axes[0, 1]
    ax.set_title('B · Visibilidade das mensagens no servidor', weight='bold', pad=15)
    ax.set_xlim(-.5, 3.5); ax.set_ylim(-.5, 1.5)
    ax.set_xticks(range(4), labels)
    ax.set_yticks([0, 1], ['Apenas agregado¹', 'Individual acessível'])
    for i, y in enumerate([1, 0, 1, 0]):
        ax.scatter(i, y, s=200, color=colors[i], zorder=3)
        ax.annotate(['Sem ruído LDP', 'Sem ruído LDP', 'Com ruído LDP', 'Com ruído LDP'][i], (i,y), xytext=(0,18), textcoords='offset points', ha='center', fontsize=9)
    ax.grid(axis='y', alpha=.2)
    # Read the selected condition's public configuration; cumulative curves below
    # represent exactly five completed cycles, excluding other historical attempts.
    design = json.loads((ROOT/'artifacts/ldp_logits/closed_loop/design.json').read_text())
    assert design['epsilon_total'] == 4 and design['delta_total_gaussian'] == 1e-5
    assert (ROOT/'artifacts/ldp_sa/status.json').exists()
    for ax, parameter, unit in [(axes[1,0], 4., 'ε'), (axes[1,1], 1e-5, 'δ')]:
        x = np.arange(1,6); y = parameter*x
        ax.plot(x, y, color=colors[2], marker='o', lw=2.5, label='Só LDP (P2)')
        ax.plot(x, y, color=colors[3], marker='x', ms=10, linestyle='--', lw=2, label='SA + LDP (P2)')
        ax.set_title(('C' if unit=='ε' else 'D') + f' · Orçamento {unit} composto por cliente²', weight='bold', pad=15)
        ax.set_xticks(x); ax.set_xlabel('Número de ciclos / seeds divulgados')
        ax.set_ylabel(f'{unit} acumulado · composição aditiva')
        ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.legend(fontsize=9)
        if unit=='δ': ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.annotate('Curvas coincidentes: SA mantém o orçamento LDP', (3,parameter*3), xytext=(1.1, parameter*4.7), fontsize=9, arrowprops={'arrowstyle':'->','color':'#526070'})
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Soft labels — privacidade nas quatro configurações', fontsize=19, weight='bold', y=.98)
    fig.text(.5,.925,'Garantias configuradas e exposição no protocolo; não são resultados de ataques nem uma pontuação de privacidade',ha='center',fontsize=11)
    fig.text(.06,.025,
             '¹ SA: oculta contribuições individuais nas hipóteses da simulação com servidor semi-honesto; limiar 12/22, 24 bits.\n'
             '² LDP: mensagem preparada, ε=4 e δ=10⁻⁵ por cliente/ciclo. Não se demonstra DP de ponta a ponta do ECG bruto.\n'
             'Sem LDP: sem garantia DP estabelecida (não ε=0). Curvas: cinco ciclos por condição; excluem tentativas e divulgações anteriores.\n'
             'Divulgar as duas condições privadas exige compor ambas. SA não reduz ε/δ e não impede envenenamento.',
             fontsize=10, linespacing=1.5)
    fig.subplots_adjust(top=.82,bottom=.22,hspace=.55,wspace=.37,left=.13,right=.97)
    for ext in ['png','pdf']:
        fig.savefig(OUT/f'privacidade_quatro_configuracoes.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)


def update_notebook():
    path=ROOT/'Privacy Techniques/LDP + SA/experiencias_ldp_sa.ipynb'
    nb=nbformat.read(path,as_version=4)
    marker='softlabels_privacy_panels'
    nb.cells=[c for c in nb.cells if c.metadata.get('comparison_section')!=marker]
    intro=nbformat.v4.new_markdown_cell('''## Gráficos de privacidade — garantias e exposição

**A:** presença das duas proteções, sem as somar numa pontuação artificial. **B:** acesso do servidor às mensagens individuais ou apenas ao agregado, nas hipóteses de SA; com LDP, as mensagens são previamente perturbadas. **C–D:** composição dos orçamentos ε e δ nas cinco seeds privadas, por cliente e por condição.

Só LDP e SA + LDP têm **as mesmas curvas ε/δ**: SA acrescenta ocultação das contribuições durante a agregação, sem reduzir o orçamento LDP. As configurações sem LDP não entram nestas curvas porque **não têm uma garantia DP estabelecida**; representá-las com ε=0 seria incorreto.

Estes gráficos descrevem as garantias e parâmetros do protocolo. Não foram medidos sucesso de ataques de inferência, reconstrução ou fuga de informação; não se apresentam esses resultados como se tivessem sido avaliados. A composição mostrada exclui execuções históricas e outras divulgações.
''')
    code=nbformat.v4.new_code_cell('''from support.softlabels_privacy_panels import main as criar_graficos_privacidade
from support.softlabels_privacy_comparison import OUT
from IPython.display import display, Image

criar_graficos_privacidade()
display(Image(filename=str(OUT / "privacidade_quatro_configuracoes.png")))
''')
    code.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((OUT/'privacidade_quatro_configuracoes.png').read_bytes()).decode()})]
    for cell in [intro,code]:
        cell.metadata['comparison_section']=marker
        nb.cells.append(cell)
    nbformat.validate(nb);nbformat.write(nb,path)


if __name__=='__main__':
    main();update_notebook()
