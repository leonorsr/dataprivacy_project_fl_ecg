"""Communication volumes from protocol logs; explicit estimates for FedAvg LDP."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/softlabels-comparison-mpl')
from pathlib import Path
import json,base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nbformat
from support.softlabels_privacy_comparison import ROOT,OUT,SOURCES


def main():
    records=[]
    for index,(condition,template) in enumerate(SOURCES.items()):
        for method in ['FedAvg','Soft labels']:
            for seed in range(2031,2036):
                estimated=method=='FedAvg' and index>=2
                folder=(ROOT/template.format(seed=seed) if method=='Soft labels' else
                    ROOT/'Privacy Techniques/SA/Communication/Fedavg/artifacts'/str(seed)/('SA' if index in [1,3] else 'plain'))
                meta=json.loads((folder/'run.json').read_text());assert meta['status']=='complete'
                f=pd.read_json(folder/'communication.json');assert len(f)==3
                assert np.allclose(f.total_bytes,f.upload_bytes+f.download_bytes)
                records.append(dict(condition=condition,method=method,seed=seed,estimated=estimated,
                    upload_bytes=int(f.upload_bytes.sum()),download_bytes=int(f.download_bytes.sum()),
                    total_bytes=int(f.total_bytes.sum()),source=str(folder.relative_to(ROOT))))
    df=pd.DataFrame(records);OUT.mkdir(parents=True,exist_ok=True)
    df.to_csv(OUT/'comunicacao_por_seed.csv',index=False)
    summary=df.groupby(['condition','method'],sort=False)[['upload_bytes','download_bytes','total_bytes']].mean()
    summary.to_csv(OUT/'comunicacao_resumo.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(12,5.2))
    x=np.arange(4);width=.32;palette=['#49677f','#238c80']
    for j,method in enumerate(['FedAvg','Soft labels']):
        rows=[df[(df.condition==c)&(df.method==method)] for c in SOURCES]
        up=np.array([r.upload_bytes.mean() for r in rows])/1024**2
        down=np.array([r.download_bytes.mean() for r in rows])/1024**2
        pos=x+(j-.5)*width
        for i in range(4):
            hatch='///' if method=='FedAvg' and i>=2 else None
            axes[0].bar(pos[i],up[i],width,color=palette[j],edgecolor='white',hatch=hatch)
            axes[0].text(pos[i],up[i]*1.12,f'{up[i]:.3f}',ha='center',fontsize=9)
            axes[1].bar(pos[i],up[i],width,color=palette[j],edgecolor='white',hatch=hatch)
            axes[1].bar(pos[i],down[i],width,bottom=up[i],color=palette[j],alpha=.32,edgecolor=palette[j],hatch=hatch)
            axes[1].text(pos[i],up[i]+down[i]+.15,f'{up[i]+down[i]:.2f}',ha='center',fontsize=9)
    axes[0].set_yscale('log');axes[0].set_ylim(.008,15)
    axes[0].set_title('a) Upload dos clientes · escala logarítmica',loc='left',weight='bold',pad=15)
    axes[1].set_title('b) Comunicação total · upload + download',loc='left',weight='bold',pad=15)
    axes[1].set_ylim(0,16)
    for ax in axes:
        ax.set_xticks(x,['Sem SA\nsem LDP','Só SA','Só LDP','SA + LDP'])
        ax.set_ylabel('MiB por ciclo de 3 rondas · 22 clientes')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Volume de comunicação: FedAvg vs. Soft labels',x=.06,ha='left',weight='bold',fontsize=16,y=.98)
    handles=[Patch(facecolor=palette[0],label='FedAvg (pesos)'),Patch(facecolor=palette[1],label='Soft labels'),Patch(facecolor='#c0c7cd',label='Download (tom claro)'),Patch(facecolor='white',edgecolor='#64717e',hatch='///',label='Estimativa: FedAvg com LDP')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.89),ncol=4,frameon=False,fontsize=9)
    fig.text(.06,.035,'Payload contabilizado pelo protocolo; inclui overhead de SA e difusão do modelo global. 1 MiB = 2²⁰ bytes.\nFedAvg com LDP: estimativa com dimensão e codificação dos pesos fixas; não houve treino privado FedAvg nestas condições.',fontsize=9,color='#45515e',linespacing=1.6)
    fig.subplots_adjust(left=.08,right=.98,top=.73,bottom=.23,wspace=.28)
    for ext in ['png','pdf','svg']:fig.savefig(OUT/f'comunicacao_fedavg_softlabels.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    caption='''**Figura — Volume de comunicação de FedAvg e Soft labels nas quatro condições.** (a) Upload agregado dos 22 clientes, em escala logarítmica. (b) Upload e download, empilhados, incluindo a distribuição do modelo global e a comunicação de SA. Valores em MiB por ciclo de três rondas, média de cinco seeds; os volumes são constantes entre seeds nas condições registadas. Contabiliza-se a difusão inicial e as difusões após cada ronda, incluindo a final uma única vez. Os valores são bytes de payload contabilizados pelo protocolo, não tráfego medido na rede; não incluem cabeçalhos de transporte, retransmissões ou energia.

**Barras tracejadas:** não existem execuções FedAvg com LDP nestes resultados. As condições Só LDP e SA + LDP de FedAvg são estimativas que reutilizam, respetivamente, os volumes sem SA e com SA, sob a hipótese de que a perturbação local mantém o número de pesos e a sua codificação. Não se aplica o mecanismo P2 de logits aos pesos nem se afirma uma garantia DP para essas estimativas. Em Soft labels usam-se os registos efetivos de P2 e P2 + SA: a margem LDP é reconvertida em 64 probabilidades antes da transmissão, sem redução da dimensão do payload.

**Interpretação:** Soft labels reduz fortemente o upload face ao envio de pesos de FedAvg. A redução na comunicação total é menor, pois ambos distribuem a mesma CNN1D global. SA acrescenta overhead nos dois métodos. Quantidade de bytes não mede quantidade semântica de informação sensível nem constitui uma métrica de privacidade. As arquiteturas e objetivos locais diferem, pelo que o gráfico compara comunicação, não equivalência de utilidade ou de custo computacional.'''
    (OUT/'comunicacao_legenda.md').write_text(caption,encoding='utf-8')
    path=ROOT/'Privacy Techniques/LDP + SA/experiencias_ldp_sa.ipynb'
    nb=nbformat.read(path,as_version=4);marker='communication_report'
    nb.cells=[c for c in nb.cells if c.metadata.get('comparison_section')!=marker]
    heading=nbformat.v4.new_markdown_cell('## Síntese de comunicação — FedAvg e Soft labels')
    code=nbformat.v4.new_code_cell('from IPython.display import display, Image\nfrom support.softlabels_privacy_comparison import OUT\ndisplay(Image(filename=str(OUT / "comunicacao_fedavg_softlabels.png")))')
    code.outputs=[nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode((OUT/'comunicacao_fedavg_softlabels.png').read_bytes()).decode()})]
    for c in [heading,code,nbformat.v4.new_markdown_cell(caption)]:c.metadata['comparison_section']=marker;nb.cells.append(c)
    nbformat.validate(nb);nbformat.write(nb,path)
    print(summary.to_string())

if __name__=='__main__':main()
