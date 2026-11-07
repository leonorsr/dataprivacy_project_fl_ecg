"""Security-focused reporting and small synthetic protocol checks; no model retraining."""
import base64,hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from support import sa_communication as core

BASE=core.BASE/'Softlabels'
RESULTS=BASE/'results/security'
TAG='softlabels_sa_security'
INTRO="""## Segurança — Soft Labels sem SA vs Soft Labels com SA

O objetivo da SA é limitar a exposição das contribuições individuais ao servidor, permitindo recuperar a soma ponderada. **Um PR-AUC semelhante não demonstra confidencialidade.** Esta secção separa as propriedades pretendidas, as verificações funcionais realizadas e as limitações não avaliadas.

Modelo de ameaça desta implementação: **servidor honesto mas curioso**, que segue o protocolo. A comparação usa a configuração A, os mesmos dados, cinco seeds, três rondas, dez épocas locais, 22 clientes e pesos de fit 72,73%/27,27%. SA: soma única de probabilidades ponderadas, 24 bits, limiar 12, novas chaves/máscaras em cada ronda, sem desistências nos treinos completos. A avaliação de desistências abaixo é isolada e sintética.

| Informação ou propriedade | Sem SA | Com SA |
|---|---|---|
| ECG privados enviados na mensagem de aprendizagem | Não | Não |
| Probabilidades individuais acessíveis em claro ao servidor | Sim | Não no fluxo previsto; os payloads recebidos estão mascarados |
| Agregado ponderado disponível | Sim | Sim, após desmascaramento |
| Identificadores, participação e contagens públicas de fit | Conhecidos | Continuam conhecidos |
| Modelo global disponível ao servidor | Sim | Sim |
| Garantia LDP / orçamento epsilon | Não | Não; SA não acrescenta uma garantia LDP |
| Proteção contra inferência a partir do agregado/modelo | Não demonstrada | Não demonstrada |

A coluna com SA descreve a visão pretendida do servidor do protocolo. A simulação corre num único processo, que possui os dados para executar clientes e avaliar os resultados. A separação entre clientes e servidor é lógica, não isolamento de processos ou dispositivos. Os artefactos privados de avaliação não são mensagens do treino. Os testes abaixo não constituem prova criptográfica nem certificação da implementação.

**Limitação da integração atual:** a função de simulação `aggregate(messages, ...)` recebe as contribuições em claro, calcula uma referência para medir o erro e só depois invoca a soma segura com clientes simulados. Portanto, estes treinos não demonstram uma ocultação end-to-end das contribuições ao processo agregador. Numa implantação real, cada cliente teria de preparar e mascarar a mensagem antes de a enviar; a referência em claro ficaria num avaliador de testes separado e de confiança.
"""

def protocol_digest():return hashlib.sha256((core.BASE.parent/'protocol.py').read_bytes()).hexdigest()

def audit():
    """Controlled artificial inputs: correctness, dropout refusal and limited integrity checks."""
    sa=core.SA;n=22;t=12;bits=24;rng=np.random.default_rng(2026)
    anomaly=rng.beta(2,2,size=(n,32));p=np.stack([1-anomaly,anomaly],axis=-1).reshape(n,64)
    weights=np.full(n,1/n);vectors={i+1:sa.quantize(weights[i]*p[i],bits) for i in range(n)}
    cases=[];example=[]
    specs=[('Sem desistências',(),(),True),('Quatro desistem antes de enviar',(1,2,17,18),(),True),
        ('Dezoito enviam; doze desmascaram',(1,2,17,18),(3,4,5,6,7,8),True),
        ('Dezoito enviam; só onze desmascaram',(1,2,17,18),(3,4,5,6,7,8,9),False)]
    for name,drop,unmask,expected in specs:
        try:
            total,report=sa.secure_sum(vectors,t,drop_masked=drop,drop_unmask=unmask)
            assert expected
            ids=report['survivor_ids'];ref=sum(vectors[u] for u in ids)%sa.MOD
            assert np.array_equal(total,ref)
            cases.append(dict(cenario=name,mensagens=n-len(drop),respostas=n-len(drop)-len(unmask),resultado='Concluiu; soma inteira exata',esperado=True))
            if not drop:
                example=[dict(cliente=u,entrada_quantizada=vectors[u][:4].tolist(),payload_mascarado=report['server_view'][u][:4].tolist()) for u in [1,2,3]]
                assert all(not np.array_equal(report['server_view'][u],vectors[u]) for u in ids)
        except RuntimeError as error:
            assert not expected
            cases.append(dict(cenario=name,mensagens=n-len(drop),respostas=n-len(drop)-len(unmask),resultado='Recusou: respostas abaixo de t',esperado=True))
    key=bytes(32);box=sa.seal(key,b'artificial share')
    altered=box[:-1]+bytes([box[-1]^1]);share_rejected=False
    try:sa.unseal(key,altered)
    except ValueError:share_rejected=True
    assert share_rejected
    # Probe the actual API: masked input vectors have no authentication check here.
    clients={u:sa.Client(u,x) for u,x in vectors.items()};server=sa.Server(t,64)
    keys=server.collect_keys({u:c.advertise() for u,c in clients.items()})
    routed=server.route({u:c.share_keys(keys,t) for u,c in clients.items()})
    masked={u:c.masked_input(routed[u]) for u,c in clients.items()}
    responses={u:c.unmask(sorted(masked),[]) for u,c in clients.items()}
    masked[1]=masked[1].copy();masked[1][0]=(int(masked[1][0])+1)%sa.MOD
    modified=server.aggregate(masked,responses);expected=sum(vectors.values())%sa.MOD
    altered_accepted=not np.array_equal(modified,expected)
    assert altered_accepted
    integrity=[dict(teste='Alteração de uma partilha cifrada',resultado='Rejeitada pelo HMAC'),
        dict(teste='Alteração de uma coordenada do vetor mascarado',resultado='Aceite pela API; agregado alterado')]
    result=dict(protocol_sha256=protocol_digest(),synthetic_inputs=True,n=n,t=t,bits=bits,
        cases=cases,example=example,integrity=integrity,share_tampering_rejected=share_rejected,
        masked_payload_tampering_accepted=altered_accepted,privacy_attacks_evaluated=False,
        scope='Functional checks, not confidentiality proof; active tampering probe is outside honest-but-curious assumptions')
    core.dump(RESULTS/'functional_audit.json',result)
    pd.DataFrame(cases).to_csv(RESULTS/'dropout_checks.csv',index=False)
    pd.DataFrame(integrity).to_csv(RESULTS/'integrity_checks.csv',index=False)
    return result

def measurements():
    frames=[];cost=[]
    for p in sorted((BASE/'artifacts').glob('*/*/run.json')):
        meta=json.loads(p.read_text());v=p.with_name('validation.json')
        if meta.get('status')!='complete' or not v.exists():continue
        assert meta['signature']==core.signature()==json.loads(v.read_text())['signature']
        f=pd.read_json(p.with_name('metrics.json'));f=f[(f['round']==3)&(f.split=='holdout')]
        for role,g in f.groupby('role'):
            ok=np.isfinite(g.pr_auc_ap)
            frames.append(dict(condition=meta['condition'],seed=meta['seed'],role=role,
                ap=np.average(g.loc[ok,'pr_auc_ap'],weights=g.loc[ok,'fit_examples'])))
        f=pd.read_json(p.with_name('communication.json'));f['condition']=meta['condition'];f['seed']=meta['seed'];cost.append(f)
    metrics=pd.DataFrame(frames);comm=pd.concat(cost,ignore_index=True)
    assert metrics.groupby(['condition','role']).seed.nunique().eq(5).all()
    assert len(comm)==30 and comm[comm.condition=='SA'].secure_integer_sum_verified.all()
    per_cycle=comm.groupby(['condition','seed'])[['upload_bytes','download_bytes','total_bytes','aggregation_seconds']].sum().reset_index()
    rows=[]
    for role,label in [('local','PR-AUC local'),('global','PR-AUC global')]:
        paired=metrics[metrics.role==role].pivot(index='seed',columns='condition',values='ap')
        rows.append(dict(metrica=label,sem_SA=paired.plain.mean(),com_SA=paired.SA.mean(),diferenca=paired.SA.mean()-paired.plain.mean(),unidade='AP; média de cinco seeds'))
    for col,label,denominator,unit in [('upload_bytes','Upload por cliente',22*1024,'KiB/cliente/ciclo'),
        ('download_bytes','Download por cliente',22*1024,'KiB/cliente/ciclo'),('total_bytes','Comunicação total por cliente',22*1024,'KiB/cliente/ciclo'),
        ('aggregation_seconds','Tempo de agregação acumulado',1,'s/ciclo; simulação local')]:
        m=per_cycle.groupby('condition')[col].mean()/denominator
        rows.append(dict(metrica=label,sem_SA=m.plain,com_SA=m.SA,diferenca=m.SA-m.plain,unidade=unit))
    comparison=pd.DataFrame(rows)
    numeric=comm.groupby('condition').agg(rondas=('round','size'),erro_medio=('aggregate_mae','mean'),maior_erro=('aggregate_max_error','max')).reset_index()
    RESULTS.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(RESULTS/'paired_comparison.csv',index=False);numeric.to_csv(RESULTS/'aggregation_error.csv',index=False)
    return comparison,numeric

def threat_comparison():
    """Categorical evidence map. No invented security percentages or attack scores."""
    result=json.loads((RESULTS/'functional_audit.json').read_text())
    assert result['protocol_sha256']==protocol_digest()
    assert result['share_tampering_rejected'] and result['masked_payload_tampering_accepted']
    rows=[
        ('Referência: servidor segue o protocolo\n(visão lógica dos payloads)',
         'Probabilidades individuais\nem claro','exposure','Vetores mascarados\nno protocolo simulado','limited'),
        ('Processo agregador malicioso\ninspeciona a memória da simulação',
         'Contribuições em claro\nna simulação','exposure','Avaliador recebe contribuições\nem claro antes de simular SA','exposure'),
        ('Atacante altera uma mensagem\nde contribuição',
         'Este código não autentica\na contribuição em claro','limited','Vetor mascarado adulterado\naceite; agregado alterado (teste)','exposure'),
        ('Atacante altera uma partilha\nde segredo cifrada',
         'Não há partilhas SA','na','Alteração rejeitada pelo HMAC\n(teste funcional)','observed'),
        ('Cliente malicioso envia previsões\nforjadas / poisoning',
         'Sem agregação robusta;\nefeito do ataque não medido','limited','SA não valida a honestidade\ndas previsões; ataque não medido','limited'),
        ('Servidor distribui modelos diferentes\nou tenta isolar um cliente',
         'Ataque não avaliado','unknown','Ataque não avaliado;\nsem garantia demonstrada','unknown'),
        ('Servidor e clientes entram\nem conluio',
         'Cenário não avaliado','unknown','Cenário não avaliado; t=12\nnão garante resistir a conluio','unknown'),
        ('Atacante tenta inferir dados a partir\ndo agregado ou do modelo global',
         'Sucesso do ataque\nnão medido; sem LDP','unknown','Sucesso do ataque\nnão medido; sem LDP','unknown'),
        ('Abandono de participantes: ficam\nmenos de 12 respostas',
         'Não há limiar de\nrecuperação SA','na','Agregação recusada\n(teste funcional; não prova anti-DoS)','observed')]
    frame=pd.DataFrame(rows,columns=['cenario','sem_SA','evidencia_sem_SA','com_SA','evidencia_com_SA'])
    frame.to_csv(RESULTS/'malicious_threat_comparison.csv',index=False)
    return frame

def threat_figure():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch,Patch
    frame=threat_comparison()
    palette={'observed':('#cde7db','#205c43'),'limited':('#fff0c7','#775716'),
        'exposure':('#f6d6d1','#8a3128'),'unknown':('#e4e8ef','#384454'),'na':('#f1f2f4','#676c74')}
    fig=plt.figure(figsize=(14,11.6),facecolor='white')
    fig.text(.045,.962,'Soft Labels com e sem SA — cenários maliciosos',fontsize=21,fontweight='bold',color='#1d2b3c')
    fig.text(.045,.933,'Matriz qualitativa de evidências do protótipo atual · não é uma pontuação de segurança',fontsize=12,color='#475569')
    ax=fig.add_axes([.035,.215,.93,.68]);ax.set_xlim(0,1);ax.set_ylim(0,len(frame)+1);ax.axis('off')
    ax.text(.012,len(frame)+.45,'Cenário / ação do atacante',fontsize=12,fontweight='bold')
    ax.text(.59,len(frame)+.45,'SEM SA',fontsize=13,fontweight='bold',ha='center')
    ax.text(.85,len(frame)+.45,'COM SA — protótipo',fontsize=13,fontweight='bold',ha='center')
    for i,row in enumerate(frame.itertuples()):
        y=len(frame)-i-.5
        ax.text(.012,y,row.cenario,va='center',fontsize=10.8,linespacing=1.5,color='#263445')
        for x,key in [(.47,'sem_SA'),(.73,'com_SA')]:
            state=getattr(row,'evidencia_'+key);background,color=palette[state]
            box=FancyBboxPatch((x,y-.39),.25,.78,boxstyle='round,pad=0.004,rounding_size=0.025',
                facecolor=background,edgecolor='white',linewidth=1.5)
            ax.add_patch(box);ax.text(x+.125,y,getattr(row,key),ha='center',va='center',fontsize=10,color=color,linespacing=1.35)
    legend=[('observed','Proteção funcional observada'),('limited','Proteção limitada / limite do desenho'),
        ('exposure','Exposição ou falha verificada'),('unknown','Ataque não avaliado'),('na','Não aplicável')]
    fig.legend(handles=[Patch(facecolor=palette[k][0],label=label) for k,label in legend],loc='lower center',
        bbox_to_anchor=(.5,.14),ncol=3,frameon=False,fontsize=10.5)
    fig.text(.045,.112,'LEITURA: verde refere-se apenas ao teste indicado; não certifica segurança contra adversários maliciosos.',fontsize=10.5,fontweight='bold',color='#263445')
    fig.text(.045,.091,'Os treinos assumem servidor honesto mas curioso. As sondagens ativas usam dados artificiais, não ataques aos ECG.\n'
        'A integração atual é uma simulação num único processo; o avaliador conhece as contribuições. Não há isolamento end-to-end.\n'
        'O agregado e os metadados continuam disponíveis. SA não equivale a LDP, não elimina poisoning e não garante disponibilidade.',fontsize=10,color='#475569',linespacing=1.5,va='top')
    fig.text(.045,.02,'Base: código e verificações do projeto. Parâmetros SA: 22 clientes, limiar 12, quantização 24 bits. Não inclui benchmark de inferência/reconstrução.',fontsize=9,color='#64748b')
    fig.savefig(RESULTS/'malicious_threat_comparison.png',dpi=170,bbox_inches='tight')
    fig.savefig(RESULTS/'malicious_threat_comparison.svg',bbox_inches='tight')
    return fig

def render():
    from IPython.display import display,Markdown
    display(Markdown('### Utilidade e custo da proteção — treinos completos'))
    comparison,numeric=measurements();display(comparison.round(6));display(numeric)
    display(Markdown('As diferenças de AP são com SA menos sem SA, nas mesmas cinco seeds. '
        'A condição SA inclui quantização a 24 bits: o efeito observado no treino não deve ser atribuído só à criptografia. '
        'Nas 15 rondas de treino com SA, foi verificado que a soma inteira segura coincide com a soma dos inteiros quantizados. '
        'O erro em probabilidades é medido separadamente contra a referência sem quantização. Bytes são payloads lógicos, sem TCP/TLS; '
        'o tempo inclui preparação e geração de chaves, sem rede real. RAM e energia no wearable não foram medidas.'))
    path=RESULTS/'functional_audit.json'
    if not path.exists():
        display(Markdown('Testes funcionais sintéticos ainda não executados. Correr `python -m support.sa_softlabels_security`.'));return
    result=json.loads(path.read_text());assert result['protocol_sha256']==protocol_digest(),'Protocol changed: rerun security audit'
    display(Markdown('### Verificações funcionais isoladas — dados artificiais\n\n'
        'Usam-se 22 vetores artificiais de 64 probabilidades, pesos 1/22, 24 bits e limiar 12. '
        'Não são novos treinos, ataques aos ECG ou resultados de privacidade sobre os pacientes.'))
    display(pd.DataFrame(result['cases']))
    display(Markdown('Exemplo das primeiras quatro coordenadas de três clientes: os payloads mascarados diferem dos valores quantizados originais. '
        '**Esta diferença visual não prova confidencialidade.** `Server.view` regista os vetores mascarados, não toda a informação do protocolo; '
        'o servidor também recebe chaves públicas, partilhas cifradas e respostas de desmascaramento previstas.'))
    display(pd.DataFrame(result['example']))
    display(Markdown('### Integridade: proteção limitada observada'))
    display(pd.DataFrame(result['integrity']))
    display(Markdown('**Limitação verificada:** o HMAC rejeita alterações nas partilhas cifradas, mas esta API não verifica a autenticidade '
        'dos vetores de contribuição mascarados. Alterar uma coordenada foi aceite e alterou o agregado. '
        'O teste é ativo e está fora da hipótese de execução honesta; não invalida a soma correta nos testes sem manipulação, '
        'mas impede afirmar proteção geral contra mensagens adulteradas. Não foi simulado um transporte autenticado.'))
    display(Markdown('### Comparação visual sob ameaças maliciosas\n\n'
        'A imagem distingue exposição verificada, proteção limitada, proteção funcional observada e ataques não avaliados. '
        'Não se atribuem níveis numéricos de segurança a partir de testes de correção. '
        'A primeira linha é uma referência com servidor cumpridor; as restantes exploram ameaças e limites do protótipo.'))
    import matplotlib.pyplot as plt
    fig=threat_figure();display(fig);plt.close(fig)
    display(Markdown('### Conclusão de segurança\n\n'
        'Comparativamente à transmissão em claro, **SA limita a exposição das probabilidades individuais no fluxo previsto**, '
        'mantendo disponível o agregado e acrescentando custo de comunicação/cálculo. A correção da soma e a recusa abaixo do limiar '
        'foram verificadas funcionalmente; não demonstram por si só uma garantia criptográfica da implementação. '
        '**Não foram medidos ataques de reconstrução, inferência de pertença/propriedades, conluio ou isolamento de clientes.** '
        'O limiar 12 é de recuperação; não significa automaticamente tolerar 11 clientes em conluio. '
        'O agregado, a participação e o modelo global continuam disponíveis e podem revelar informação; '
        'SA não oferece LDP nem elimina poisoning. A integridade dos vetores e cenários com servidor malicioso requerem mecanismos e avaliação adicionais.\n\n'
        'Referências: [Bonawitz et al., 2017 — Secure Aggregation](https://research.google/pubs/practical-secure-aggregation-for-privacy-preserving-machine-learning/); '
        '[Pasquini et al. — limites perante servidor malicioso](https://arxiv.org/abs/2111.07380).'))

def append_section(notebook):
    import IPython.display
    notebook.cells=[c for c in notebook.cells if TAG not in c.metadata.get('tags',[])]
    outputs=[]
    def capture(value,**kwargs):
        if hasattr(value,'savefig'):
            buffer=io.BytesIO();value.savefig(buffer,format='png',dpi=130,bbox_inches='tight')
            data={'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':'Soft Labels com e sem SA — cenários maliciosos'}
        elif hasattr(value,'_repr_markdown_'):data={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        elif hasattr(value,'to_html'):data={'text/html':value.to_html(index=False),'text/plain':repr(value)}
        else:data={'text/plain':repr(value)}
        outputs.append(nbformat.v4.new_output('display_data',data=data))
    original=IPython.display.display;IPython.display.display=capture
    try:render()
    finally:IPython.display.display=original
    notebook.cells.extend([nbformat.v4.new_markdown_cell(INTRO,metadata={'tags':[TAG]}),
        nbformat.v4.new_code_cell('from support.sa_softlabels_security import render\nrender()',execution_count=2,outputs=outputs,metadata={'tags':[TAG]})])
    return notebook

def refresh():
    path=BASE/'experiencias_softlabels.ipynb';notebook=nbformat.read(path,4)
    append_section(notebook);nbformat.validate(notebook);nbformat.write(notebook,path)

if __name__=='__main__':audit();refresh()
