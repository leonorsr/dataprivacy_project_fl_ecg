from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/softlabels-comparison-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import nbformat
import base64
root=Path.cwd();out=root/'artifacts/softlabels_overview/comparacao_privacidade'
f=pd.read_csv(out/'por_seed.csv')
conditions=['Sem SA / sem LDP','Só SA','Só LDP (P2)','SA + LDP (P2)']
labels=['Sem SA e sem LDP','Só SA','Só LDP','SA + LDP']
colors=['#637487','#25877f','#c58b31','#7856a0']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
fig=plt.figure(figsize=(10.5,5.3))
gs=fig.add_gridspec(1,3,width_ratios=[2.2,2.2,1.25],wspace=.5)
y=np.arange(4)
for j,role in enumerate(['local','global']):
 ax=fig.add_subplot(gs[0,j])
 for i,c in enumerate(conditions):
  v=f[(f.condition==c)&(f.role==role)].pr_auc_ap
  ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='o',ms=8,color=colors[i],capsize=4,lw=1.6)
  ax.text(v.mean(),i-.22,f'{v.mean():.3f}',ha='center',fontsize=9,color=colors[i],weight='bold')
 ax.set_xlim(0,1);ax.set_ylim(3.55,-.6)
 ax.set_yticks(y,labels if j==0 else ['']*4)
 ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel('PR-AUC (AP)')
 ax.set_title(('a) Modelos locais' if j==0 else 'b) Modelo global'),loc='left',fontsize=11,weight='bold',pad=18)
 ax.grid(axis='x',color='#e5e8ec',lw=.8);ax.set_axisbelow(True)
 ax.spines[['top','right','left']].set_visible(False);ax.spines['bottom'].set_color('#b7bec5')
 ax.tick_params(axis='y',length=0,pad=10)
 ax.axhspan(2.6,3.4,color='#7856a0',alpha=.055,zorder=0)
ax=fig.add_subplot(gs[0,2])
m=np.array([[0,0],[1,0],[0,1],[1,1]])
ax.imshow(m,cmap=ListedColormap(['#f0f1f3','#d9eae4']),vmin=0,vmax=1,aspect='auto',extent=(-.5,1.5,3.5,-.5))
for r in range(4):
 for c in range(2):ax.text(c,r,'Sim' if m[r,c] else '—',ha='center',va='center',fontsize=11,color='#18584d' if m[r,c] else '#858d96')
ax.set_ylim(3.55,-.6);ax.set_xticks([0,1],['SA','LDP']);ax.set_yticks([])
ax.set_title('c) Proteção',loc='left',fontsize=11,weight='bold',pad=18)
ax.tick_params(length=0);ax.spines[:].set_visible(False)
fig.suptitle('Utilidade e proteção das contribuições em Soft labels',x=.04,ha='left',fontsize=15,weight='bold',y=.97)
fig.text(.04,.085,'Pontos: média de cinco seeds; barras: ±1 desvio padrão. Mesmos holdouts, ronda 3.',fontsize=9,color='#46515d')
fig.text(.04,.04,'SA + LDP preserva 99,63% da AP local de referência e combina as duas proteções.',fontsize=10,color='#533d70',weight='bold')
fig.subplots_adjust(left=.19,right=.97,top=.78,bottom=.23)
for ext in ['png','pdf','svg']:fig.savefig(out/f'resumo_relatorio.{ext}',dpi=300,bbox_inches='tight')
plt.close(fig)
caption='''**Figura — Utilidade e proteção das contribuições nas quatro configurações de Soft labels.** (a–b) PR-AUC (AP) local e global na ronda 3, nos mesmos holdouts dos 22 clientes de treino. Cada ponto representa a média de cinco seeds (2031–2035); as barras representam ±1 desvio padrão amostral entre seeds. Em cada seed, calcula-se a média das métricas por cliente ponderada pelo número de exemplos de fit; não se trata de AP pooled nem de resultados de test. (c) Presença dos mecanismos de proteção: SA oculta contribuições individuais durante a agregação nas hipóteses da simulação com servidor semi-honesto; LDP perturba a mensagem preparada. “Sim” indica a presença do mecanismo, não uma medição de eficácia contra ataques. LDP P2 usa mecanismo Gaussiano, margem, C=0,5, ε=4 e δ=10⁻⁵ por cliente/ciclo; SA usa 24 bits e limiar 12/22. As condições com LDP têm o mesmo orçamento: SA não reduz ε/δ. A composição de cinco seeds por condição privada corresponde a ε=20 e δ=5×10⁻⁵, além de outras divulgações. O ruído independente entre condições privadas impede isolar o efeito da SA; as diferenças observadas não demonstram equivalência estatística. A garantia LDP refere-se à mensagem preparada, sem demonstração de privacidade de ponta a ponta do ECG bruto.'''
(out/'legenda_relatorio.md').write_text(caption,encoding='utf-8')
path=root/'Privacy Techniques/LDP + SA/experiencias_ldp_sa.ipynb';nb=nbformat.read(path,as_version=4)
marker='resumo_relatorio';nb.cells=[c for c in nb.cells if c.metadata.get('comparison_section')!=marker]
heading=nbformat.v4.new_markdown_cell('## Figura de síntese para o relatório')
code=nbformat.v4.new_code_cell('from IPython.display import display, Image\nfrom support.softlabels_privacy_comparison import OUT\ndisplay(Image(filename=str(OUT / "resumo_relatorio.png")))')
code.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((out/'resumo_relatorio.png').read_bytes()).decode()})]
for c in [heading,code,nbformat.v4.new_markdown_cell(caption)]:c.metadata['comparison_section']=marker;nb.cells.append(c)
nbformat.validate(nb);nbformat.write(nb,path)
print('Figura PNG/PDF/SVG e legenda guardadas; notebook atualizado.')
