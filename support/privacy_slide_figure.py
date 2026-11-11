"""Slide-ready privacy/utility operating point, without inventing a privacy score."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/softlabels-comparison-mpl')
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import nbformat
from support.softlabels_privacy_comparison import ROOT,OUT


def main():
 f=pd.read_csv(OUT/'por_seed.csv');q=f[f.role=='local']
 base=q[q.condition=='Sem SA / sem LDP'].pr_auc_ap
 private=q[q.condition=='SA + LDP (P2)'].pr_auc_ap
 loss=base.mean()-private.mean()
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42})
 fig=plt.figure(figsize=(16,9),facecolor='#f6f8fb')
 fig.text(.055,.91,'Privacidade × utilidade',fontsize=30,weight='bold',color='#172b40')
 fig.text(.055,.858,'Configuração final · Soft labels + LDP P2 + Secure Aggregation',fontsize=17,color='#56677a')
 for x,w in [(.045,.28),(.345,.36),(.725,.23)]:
  fig.add_artist(FancyBboxPatch((x,.27),w,.52,boxstyle='round,pad=0.012,rounding_size=0.015',transform=fig.transFigure,facecolor='white',edgecolor='#e1e7ef',zorder=0))
 fig.text(.065,.735,'GARANTIA DE PRIVACIDADE',fontsize=12,weight='bold',color='#95651c')
 fig.text(.065,.645,'ε = 4',fontsize=37,weight='bold',color='#95651c')
 fig.text(.065,.59,'δ = 10⁻⁵',fontsize=25,color='#95651c')
 fig.text(.065,.535,'por cliente / ciclo de 3 rondas',fontsize=13,color='#34485e')
 fig.text(.065,.46,'LDP P2 · Gaussiano\nMargem binária · clipping C = 0,5',fontsize=13,color='#34485e',linespacing=1.8)
 fig.text(.065,.325,'Âmbito: mensagem preparada;\nsem garantia de ponta a ponta\ndo ECG bruto.',fontsize=12,color='#64748b',linespacing=1.6)
 fig.text(.365,.735,'UTILIDADE LOCAL OBSERVADA',fontsize=12,weight='bold',color='#6f4d96')
 ax=fig.add_axes([.39,.43,.27,.23],facecolor='white')
 for i,(v,c) in enumerate([(base,'#7b8998'),(private,'#7856a0')]):
  ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='o',ms=10,capsize=5,color=c,lw=2)
  ax.text(v.mean(),i+.24,f'{v.mean():.4f}',ha='center',color=c,fontsize=13,weight='bold')
 ax.set_yticks([0,1],['Referência','LDP + SA']);ax.set_ylim(-.55,1.6);ax.set_xlim(.74,.86)
 ax.set_xticks([.74,.78,.82,.86]);ax.set_xlabel('PR-AUC (AP)',fontsize=11)
 ax.grid(axis='x',alpha=.18);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,labelsize=11)
 fig.text(.365,.34,f'−{loss:.4f} de AP local média'.replace('.',','),fontsize=19,weight='bold',color='#6f4d96')
 fig.text(.365,.292,'Média ± desvio padrão · 5 seeds · holdout, ronda 3',fontsize=10,color='#64748b')
 fig.text(.745,.735,'PROTEÇÃO ADICIONAL',fontsize=12,weight='bold',color='#187f72')
 fig.text(.745,.64,'SA',fontsize=37,weight='bold',color='#187f72')
 fig.text(.745,.54,'Oculta contribuições\nindividuais na agregação¹',fontsize=14,color='#34485e',linespacing=1.6)
 fig.text(.745,.43,'24 bits · limiar 12/22',fontsize=13,color='#34485e')
 fig.text(.745,.33,'Mantém ε, δ\ne o ruído LDP.',fontsize=14,weight='bold',color='#187f72',linespacing=1.5)
 fig.text(.055,.185,'Um ponto de operação: garantia LDP explícita e pequena redução média de utilidade local.',fontsize=17,weight='bold',color='#263d55')
 fig.text(.055,.11,'Cinco seeds divulgadas compõem ε = 20 e δ = 5 × 10⁻⁵ por condição, além de outras divulgações.\n¹ SA: simulação com servidor semi-honesto. Diferenças de AP sem demonstração de equivalência estatística.',fontsize=11,color='#64748b',linespacing=1.7)
 OUT.mkdir(parents=True,exist_ok=True)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'slide_privacidade_utilidade.{ext}',dpi=200,facecolor=fig.get_facecolor())
 plt.close(fig)
 path=ROOT/'Privacy Techniques/LDP + SA/experiencias_ldp_sa.ipynb';nb=nbformat.read(path,as_version=4);marker='privacy_utility_slide'
 nb.cells=[c for c in nb.cells if c.metadata.get('comparison_section')!=marker]
 heading=nbformat.v4.new_markdown_cell('## Slide 2 — privacidade e utilidade da configuração final\n\nA figura apresenta o ponto de operação escolhido, não uma curva de desempenho para vários valores de ε. A referência é Soft labels sem SA nem LDP. As barras representam o desvio padrão entre seeds, não intervalos de confiança. A comparação não isola causalmente o efeito de cada mecanismo.')
 code=nbformat.v4.new_code_cell('from IPython.display import display, Image\nfrom support.softlabels_privacy_comparison import OUT\ndisplay(Image(filename=str(OUT / "slide_privacidade_utilidade.png")))')
 code.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((OUT/'slide_privacidade_utilidade.png').read_bytes()).decode()})]
 for c in [heading,code]:c.metadata['comparison_section']=marker
 # Immediately after the final summary's image, before its explanatory caption.
 indices=[i for i,c in enumerate(nb.cells) if c.metadata.get('comparison_section')=='final_round_summary' and c.cell_type=='code']
 at=indices[-1]+1 if indices else len(nb.cells)
 nb.cells[at:at]=[heading,code];nbformat.validate(nb);nbformat.write(nb,path)
 print('Imagem incorporada após a síntese final; PNG/PDF/SVG exportados.')

if __name__=='__main__':main()
