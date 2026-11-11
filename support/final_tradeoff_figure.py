"""Report summary: utility, configured protection and upload per round."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/softlabels-comparison-mpl')
import base64,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import nbformat
from support.softlabels_privacy_comparison import ROOT,OUT,SOURCES


def main():
 f=pd.read_csv(OUT/'por_seed.csv');comm=pd.read_csv(OUT/'comunicacao_por_seed.csv')
 baseline=[]
 for seed in range(2031,2036):
  folder=ROOT/'Privacy Techniques/SA/Communication/Fedavg/artifacts'/str(seed)/'plain'
  meta=json.loads((folder/'run.json').read_text());assert meta['status']=='complete'
  m=pd.read_json(folder/'metrics.json')
  for role in ['local','global']:
   q=m[(m['round']==3)&(m.role==role)&(m.split=='holdout')]
   assert len(q)==22 and q.fit_examples.sum()==2816
   reference=json.loads((ROOT/SOURCES['Sem SA / sem LDP'].format(seed=seed)/'run.json').read_text())
   assert meta['initial_data_registry']==reference['initial_data_registry']
   baseline.append(dict(condition='FedAvg baseline',seed=seed,role=role,pr_auc_ap=np.average(q.pr_auc_ap,weights=q.fit_examples)))
 f=pd.concat([pd.DataFrame(baseline),f],ignore_index=True)
 fed=comm[(comm.method=='FedAvg')&(~comm.estimated)&(comm.condition=='Sem SA / sem LDP')].copy()
 fed['condition']='FedAvg baseline';comm=pd.concat([fed,comm],ignore_index=True)
 conditions=['FedAvg baseline',*SOURCES];labels=['FedAvg (baseline)','Sem SA e sem LDP','Só SA','Só LDP (P2)','SA + LDP (P2)']
 colors=['#354354','#637487','#25877f','#c58b31','#7856a0'];y=np.arange(5)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
 fig=plt.figure(figsize=(13,5.8));gs=fig.add_gridspec(1,4,width_ratios=[1.5,1.5,1.1,1.8],wspace=.4)
 records=[]
 for j,role in enumerate(['local','global']):
  ax=fig.add_subplot(gs[0,j])
  for i,c in enumerate(conditions):
   v=f[(f.condition==c)&(f.role==role)].pr_auc_ap
   ax.errorbar(v.mean(),i,xerr=v.std(ddof=1),fmt='o',ms=7,color=colors[i],capsize=3,lw=1.4)
   ax.text(v.mean(),i-.22,f'{v.mean():.3f}',ha='center',fontsize=9,color=colors[i],weight='bold')
  ax.set_xlim(0,1);ax.set_xticks([0,.5,1]);ax.set_xlabel('PR-AUC (AP) ↑')
  ax.set_title(['a) Utilidade local','b) Utilidade global'][j],loc='left',fontsize=10,weight='bold',pad=18)
  ax.set_yticks(y,labels if j==0 else ['']*5)
  ax.grid(axis='x',color='#e5e8ec');ax.set_axisbelow(True)
  ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=10)
  ax.set_ylim(4.55,-.6);ax.axhspan(3.6,4.4,color=colors[4],alpha=.055,zorder=0)
 ax=fig.add_subplot(gs[0,2]);m=np.array([[0,0],[0,0],[1,0],[0,1],[1,1]])
 ax.imshow(m,cmap=ListedColormap(['#f0f1f3','#d9eae4']),vmin=0,vmax=1,aspect='auto',extent=(-.5,1.5,4.5,-.5))
 for r in range(5):
  for c in range(2):ax.text(c,r,'Sim' if m[r,c] else '—',ha='center',va='center',fontsize=10,color='#18584d' if m[r,c] else '#858d96')
 ax.set_ylim(4.55,-.6);ax.set_xticks([0,1],['SA','LDP']);ax.set_yticks([]);ax.tick_params(length=0);ax.spines[:].set_visible(False)
 ax.set_title('c) Proteção¹',loc='left',fontsize=10,weight='bold',pad=18)
 ax=fig.add_subplot(gs[0,3])
 for i,c in enumerate(conditions):
  q=comm[(comm.method==('FedAvg' if c=='FedAvg baseline' else 'Soft labels'))&(comm.condition==c)]
  value=q.upload_bytes.mean()/3/22/1024
  ax.barh(i,value,height=.42,color=colors[i])
  ax.text(value*1.15,i,f'{value:.3f}',va='center',fontsize=10,weight='bold',color=colors[i])
  records.append(dict(condition=c,upload_KiB_client_round=value,upload_KiB_all_clients_round=q.upload_bytes.mean()/3/1024))
 ax.set_xlim(.1,160);ax.set_xscale('log');ax.set_ylim(4.55,-.6);ax.set_yticks([])
 ax.set_xlabel('KiB / cliente / ronda ↓');ax.set_title('d) Upload² (escala log)',loc='left',fontsize=10,weight='bold',pad=18)
 ax.grid(axis='x',color='#e5e8ec');ax.set_axisbelow(True);ax.spines[['top','right','left']].set_visible(False)
 ax.axhspan(3.6,4.4,color=colors[4],alpha=.055,zorder=0)
 fig.suptitle('FedAvg e Soft labels: utilidade, proteção e comunicação',x=.035,ha='left',fontsize=16,weight='bold',y=.98)
 fig.text(.035,.1,'Utilidade: média ± desvio padrão de cinco seeds, ronda 3, mesmos holdouts. ¹ Garantias do protocolo, não eficácia medida de ataques.',fontsize=9,color='#46515d')
 fig.text(.035,.057,'² Upload médio dos clientes, incluindo overhead de SA; exclui download. 1 KiB = 1024 bytes. Volume de bytes não mede fuga de informação.',fontsize=9,color='#46515d')
 fig.subplots_adjust(left=.16,right=.98,top=.78,bottom=.24)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'sintese_final_por_ronda.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig);pd.DataFrame(records).to_csv(OUT/'sintese_upload_por_ronda.csv',index=False)
 caption='''**Baseline FedAvg:** a primeira linha usa os registos de FedAvg sem SA nem LDP, nos mesmos clientes, holdouts, seeds e ronda final. O upload envia pesos; as restantes linhas enviam soft labels. As arquiteturas e objetivos de treino locais diferem entre métodos. O painel de upload usa escala logarítmica para acomodar a diferença de volume.

**Figura — Síntese de utilidade, proteção e volume enviado por ronda nas quatro configurações de Soft labels.** (a–b) PR-AUC local e global: média ± desvio padrão amostral entre cinco seeds na ronda 3; métricas por cliente ponderadas pelos exemplos de fit, nos mesmos holdouts dos 22 clientes de treino. Não são métricas de test nem AP pooled. (c) Presença das proteções SA e LDP, sem pontuação artificial de privacidade. SA oculta contribuições durante a agregação nas hipóteses da simulação com servidor semi-honesto, com 24 bits e limiar 12/22. LDP P2 perturba a mensagem preparada, com mecanismo Gaussiano, margem, C=0,5, ε=4 e δ=10⁻⁵ por cliente/ciclo; o orçamento é igual com ou sem SA e a composição das cinco seeds é ε=20, δ=5×10⁻⁵ por condição privada, além das restantes divulgações. Não se demonstra privacidade de ponta a ponta do ECG bruto nem resistência a envenenamento.

(d) Upload médio por cliente e por ronda, calculado a partir dos registos das três rondas e cinco seeds. Os volumes são constantes entre rondas nas condições registadas. Inclui a contribuição e as mensagens de SA contabilizadas no upload; exclui download, cabeçalhos de rede e retransmissões. Sem SA: **0,254 KiB/cliente/ronda** (**5,586 KiB/ronda** no conjunto dos 22 clientes). Com SA: **5,180 KiB/cliente/ronda** (**113,953 KiB/ronda** no conjunto). LDP mantém o volume transmitido neste protocolo, que envia 64 probabilidades após reconversão das margens. Os bytes medem volume de comunicação, não informação sensível revelada.

**Leitura do compromisso:** SA + LDP combina as duas proteções com AP local média de 0,7989, próxima de 0,8018 na referência, mas aumenta o upload devido ao overhead de SA. A utilidade global é inferior à local e variável entre seeds. Os resultados não demonstram equivalência estatística; o ruído independente entre condições privadas impede isolar o efeito de SA. A adequação a wearables exige ainda medir energia, latência e desempenho em utilização real.'''
 # Keep numerical caption in sync with the logged bytes.
 a=records[1];b=records[2]
 caption=caption.replace('5,180',f'{b["upload_KiB_client_round"]:.3f}'.replace('.',','))
 caption=caption.replace('113,953',f'{b["upload_KiB_all_clients_round"]:.3f}'.replace('.',','))
 (OUT/'sintese_final_legenda.md').write_text(caption,encoding='utf-8')
 path=ROOT/'Privacy Techniques/LDP + SA/experiencias_ldp_sa.ipynb';nb=nbformat.read(path,as_version=4);marker='final_round_summary'
 nb.cells=[c for c in nb.cells if c.metadata.get('comparison_section')!=marker]
 title=nbformat.v4.new_markdown_cell('## Síntese final — utilidade, proteção e volume enviado por ronda')
 code=nbformat.v4.new_code_cell('from IPython.display import display, Image\nfrom support.softlabels_privacy_comparison import OUT\ndisplay(Image(filename=str(OUT / "sintese_final_por_ronda.png")))')
 code.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((OUT/'sintese_final_por_ronda.png').read_bytes()).decode()})]
 for c in [title,code,nbformat.v4.new_markdown_cell(caption)]:c.metadata['comparison_section']=marker;nb.cells.append(c)
 nbformat.validate(nb);nbformat.write(nb,path)
 print(pd.DataFrame(records).to_string(index=False))

if __name__=='__main__':main()
