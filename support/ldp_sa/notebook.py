"""Reviewable combined experiment notebook with embedded incremental results."""
import io,base64,json
import nbformat
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from support.federated_global.shared import dump
from .experiment import BASE,FOLDER,DESIGN,signature,folder_for

INTRO='''# LDP + SA — distilação federada e avaliação final

## Configuração selecionada e justificação para wearables

**Escolha: LDP P2 (margem binária, C=0,5, Gaussiano, ε total=4 e δ total=10⁻⁵ por cliente/ciclo) + Secure Aggregation de soma única, 24 bits e limiar t=12 em 22 clientes.**

Na deteção de arritmias em wearables, os ECG e as contribuições dos utilizadores são sensíveis. Pretende-se preservar a qualidade da deteção local, limitar a exposição das mensagens e manter um custo de comunicação compatível com dispositivos limitados. A experiência anterior encontrou o maior PR-AUC local médio privado em P2: **0,7982**, face a 0,8018 sem LDP e 0,7964 em P1; P3 obteve 0,7895. Esta é uma seleção exploratória, anterior à abertura do test, não prova estatística de superioridade. Se δ=0 fosse obrigatório, P3 seria a opção compatível entre as testadas; se falsos alarmes fossem o objetivo principal, P1 também mereceria avaliação.

A margem preserva as probabilidades binárias antes da perturbação e utiliza 32 coordenadas no mecanismo local. Após LDP, convertem-se as margens em duas probabilidades por âncora e são estas **64 probabilidades** que entram em SA: não se assume aqui redução de comunicação para 32 valores. Os sinais ECG não são enviados. A CNN1D global completa é redistribuída; o custo do download e o consumo energético no wearable não desaparecem.

SA com soma única revela menos detalhe por origem do que somas separadas. A precisão de 24 bits segue o estudo anterior de quantização; t=12 exige pelo menos 12 respostas e permite até dez desistências, nas condições do protocolo. Isto é um compromisso operacional, não uma configuração universalmente ótima. Neste treino, os 22 clientes participam nas três rondas, sem desistências simuladas.

## Privacidade e atacantes maliciosos: âmbito real

LDP perturba localmente a mensagem antes da agregação, e SA oculta as contribuições individuais no protocolo. **Mantém-se todo o ruído LDP: não se reduz ε/ruído por adicionar SA.** Esta experiência não implementa uma nova prova de privacidade diferencial distribuída.

A implementação SA é uma simulação de investigação que assume **servidor semi-honesto**, sem isolamento de processos, autenticação de produção ou prova de resistência a servidor ativamente malicioso. Usa máscaras, X25519 e partilhas Shamir; não inclui todas as verificações necessárias para segurança maliciosa do protocolo completo. Os testes de rejeição de mensagens inválidas e adulteração de partilhas são verificações funcionais, não uma auditoria criptográfica.

**LDP + SA não impede envenenamento/backdoors:** um cliente malicioso pode enviar uma contribuição válida mas nociva ou recusar colaborar. O clipping é aplicado no código dos clientes honestos; não há prova criptográfica de que um cliente arbitrário o cumpriu. O servidor também não pode garantir a integridade do modelo distribuído contra si próprio. Para afirmar resistência a estes ataques seriam necessárias defesas e avaliação adversarial adicionais. A garantia LDP refere-se ao canal da mensagem preparada com transformações fixas; o pré-processamento histórico usa estatísticas agregadas do fit, pelo que não se demonstra privacidade de ponta a ponta do ECG bruto. Métricas, checkpoints e resultados de test não são divulgações DP.

Referências: [Bonawitz et al., Secure Aggregation](https://eprint.iacr.org/2017/281.pdf); [Bagdasaryan et al., limites perante envenenamento](https://proceedings.mlr.press/v108/bagdasaryan20a.html).

## Protocolo congelado antes do test

- Candidata A: AE+CNN no Icentia; MobileNetV2 no PhysioNet; CNN1D global com adaptadores por origem.
- Cinco seeds (2031–2035), três rondas, dez épocas locais por ronda, três passos do modelo global por ronda; 32 âncoras sintéticas sem rótulos clínicos.
- Os mesmos 22 clientes de treino: 128 exemplos de fit e 32 de holdout por cliente; pesos Icentia 72,73% e PhysioNet 27,27%.
- O objetivo principal durante o treino continua a ser PR-AUC local no holdout do próprio cliente. Sem escolher seed ou ronda por resultados de test.
- **Test final:** todos os exemplos elegíveis dos clientes reservados, sem reequilíbrio artificial: dois clientes Icentia e um PhysioNet, distintos de train/dev. Classe original 3 (ruído) excluída; 0 é normal e 1/2 são anomalia.
- O modelo global de cada seed é avaliado em test após concluir e validar os cinco treinos. Não se treinam modelos nos clientes de test e não se avaliam modelos locais de outros clientes como se fossem avaliação local. Holdout local e test global medem contextos diferentes.
- Normalização ajustada apenas ao fit de treino, ronda final fixa e decisão argmax/0,5. Sem ajuste de limiar nem seleção com test.
- Métricas de test por cliente e origem, e combinadas; pooled AP é distinguida da média de AP por cliente. O conjunto combinado é dominado pela quantidade de Icentia: consultar também cada origem.

O orçamento de cada mensagem é ε=4/3 e δ=10⁻⁵/3. As cinco seeds compõem ε=20 e δ=5×10⁻⁵ por cliente se todas as mensagens forem divulgadas, além de execuções anteriores e tentativas registadas. Não se apresenta a grelha inteira como ε=4. As duas fontes de aleatoriedade segura (ruído e chaves SA) são independentes das seeds de treino. As comparações com a LDP anterior não partilham a mesma realização de ruído e não isolam causalmente o efeito da SA.
'''

def render_results():
    from IPython.display import display,Markdown
    state=BASE/'status.json'
    display(json.loads(state.read_text()) if state.exists() else {'status':'prepared_not_started'})
    live=BASE/'runs/status.json'
    if live.exists():display(json.loads(live.read_text()))
    rows=[];comm=[]
    for seed in DESIGN['seeds']:
        folder=folder_for(seed);meta=folder/'run.json'
        if not meta.exists() or json.loads(meta.read_text()).get('status')!='complete':continue
        frame=pd.read_json(folder/'metrics.json')
        for role in ['local','global']:
            q=frame[(frame['round']==3)&(frame.role==role)&(frame.split=='holdout')]
            row=dict(seed=seed,role=role)
            for metric in ['pr_auc_ap','recall','far','f1','roc_auc']:
                row[metric]=float(np.average(q[metric],weights=q.fit_examples))
            rows.append(row)
        c=pd.read_json(folder/'communication.json');comm.append(dict(seed=seed,upload_KiB=c.upload_bytes.sum()/1024,
            download_KiB=c.download_bytes.sum()/1024,max_quantization_error=c.aggregate_max_error.max(),
            integer_sums_verified=bool(c.secure_integer_sum_verified.all())))
    display(Markdown(f'### Treino: {len(rows)//2}/5 seeds completas'))
    if rows:
        f=pd.DataFrame(rows);display(f.round(6));display(f.groupby('role')[['pr_auc_ap','recall','far','f1']].agg(['mean','std']).round(6))
        display(pd.DataFrame(comm).round(6))
        fig,ax=plt.subplots(figsize=(7,4))
        for i,role in enumerate(['local','global']):
            q=f[f.role==role].pr_auc_ap
            ax.bar(i,q.mean(),yerr=q.std() if len(q)>1 else 0,capsize=5,color=['#247b68','#4178a4'][i])
            ax.text(i,q.mean()+.05,f'{q.mean():.4f}',ha='center')
        ax.set_xticks([0,1],['Local — próprio holdout','Global — mesmos holdouts']);ax.set_ylim(0,1)
        ax.set_ylabel('PR-AUC (AP)');ax.set_title('LDP P2 + SA: média ± desvio entre seeds completas')
        buffer=io.BytesIO();fig.savefig(buffer,format='png',dpi=140,bbox_inches='tight');plt.close(fig)
        display({'image/png':base64.b64encode(buffer.getvalue()).decode()},raw=True)
    tests=BASE/'test/metrics.json';validation=BASE/'test/validation.json'
    display(Markdown('### Avaliação final em test — modelo global'))
    if tests.exists():
        final=validation.exists() and json.loads(validation.read_text()).get('status')=='complete'
        display(Markdown('**Avaliação completa e validada.**' if final else '**Avaliação em curso; resultados parciais, sem decisões de seleção.**'))
        frame=pd.read_json(tests)
        display(frame[['seed','dataset','client','scope','n_examples','pr_auc_ap','recall','far','f1']].round(6))
        q=frame[frame.scope.isin(['dataset_pooled','all_pooled'])]
        if not q.empty:display(q.groupby(['dataset','scope'])[['pr_auc_ap','recall','far','f1']].agg(['mean','std']).round(6))
        if final:display(json.loads(validation.read_text()))
    else:display(Markdown('O test ainda não foi avaliado. Só será aberto após concluir todos os treinos e congelar os checkpoints.'))
    security=BASE/'smoke_checks.json'
    if security.exists():display(json.loads(security.read_text()))
    display(Markdown('**Limite da conclusão:** estas métricas medem utilidade com participação honesta, não resistência a ataques maliciosos. '
        'A escolha foi feita em resultados exploratórios anteriores; test é avaliação final deste protocolo, não nova oportunidade de selecionar parâmetros.'))

def capture(function):
    import IPython.display
    output=[]
    def display(value,**kwargs):
        if kwargs.get('raw'):data=value
        elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False),'text/plain':str(value)}
        elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        else:data={'text/plain':str(value)}
        output.append(nbformat.v4.new_output('display_data',data=data))
    original=IPython.display.display;IPython.display.display=display
    try:function()
    finally:IPython.display.display=original
    return output

def refresh():
    FOLDER.mkdir(parents=True,exist_ok=True)
    setup="from pathlib import Path\nimport sys\nroot=Path.cwd().resolve()\nwhile not (root/'requirements.txt').exists() and root!=root.parent:root=root.parent\nsys.path.insert(0,str(root))"
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(INTRO),nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell('from support.ldp_sa.notebook import render_results\nrender_results()',execution_count=1,outputs=capture(render_results))],
        metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'frozen_design':DESIGN,'scientific_signature':signature()})
    nbformat.validate(n);path=FOLDER/'experiencias_ldp_sa.ipynb';temp=path.with_suffix('.tmp.ipynb');nbformat.write(n,temp);temp.replace(path)

if __name__=='__main__':refresh()
