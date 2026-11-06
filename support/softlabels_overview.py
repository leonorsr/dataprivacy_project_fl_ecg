"""Ordered Exp. 1–7 notebook and an exploratory, fit-weighted AP overview."""
import os
os.environ.setdefault('MPLBACKEND','Agg');os.environ.setdefault('MPLCONFIGDIR','/tmp/softlabels-overview-mpl')
import base64,io,json,hashlib,time,fcntl
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'SoftLabels tests/experiment_overview'
GRID=ROOT/'SoftLabels tests/artifacts/global_student_sample_weighted_v2'
MOMENTS=ROOT/'SoftLabels tests/moments_comparison'
PATH=ROOT/'SoftLabels tests/comparation_softlabels.ipynb'
TITLES={1:'Avaliação separada por dataset',2:'Combinação dos melhores modelos',3:'Exploração alargada de combinações',
        4:'Escolha da arquitetura por cliente',5:'Melhores combinações e referências homogéneas',
        6:'Comparação final A/B',7:'Representação das mensagens e MC Dropout'}
DESCRIPTIONS={
    1:'Cinco arquiteturas em cada dataset, em federações separadas: dez configurações, seed 2026. Cada resultado desta secção refere-se apenas à sua origem.',
    2:'Cruzamento dos top 3 de cada dataset: nove configurações mistas, seed 2026.',
    3:'União das cinco arquiteturas atribuída às duas origens: 25 configurações, seeds 2026–2028.',
    4:'Três regras de escolha: modelo comum, por dataset ou por cliente. Amostra ampliada, seleção interna sem holdout/dev; seeds 2026–2030.',
    5:'Análise focada de nove cruzamentos top 3 × top 3 e quatro referências homogéneas adicionais, 13 configurações, seeds 2026–2028. Subconjunto da grelha da Exp. 3, sem validação independente.',
    6:'A: AE+CNN no Icentia e MobileNetV2 no PhysioNet. B: CNN1D no Icentia e MobileNetV2 no PhysioNet. Amostra ampliada, dez épocas locais por ronda, seeds 2031–2035. Fonte: versão ponderada validada; os artefactos históricos 50/50 não são usados.',
    7:'Quatro representações: 64 probabilidades, 32 probabilidades de anomalia, 32 médias MC e 32 médias + 32 std MC. Vinte passagens MC Dropout por âncora, candidata A, dez épocas locais por ronda, seeds 2031–2035. Comparação sem SA/LDP; mantém-se a identidade da âncora e da classe anomalia.'}
LABELS={'common':'Modelo comum','per_dataset':'Modelo por dataset','per_client':'Modelo por cliente',
    'A_AE_CNN__MobileNetV2':'A: AE+CNN / MobileNetV2','B_CNN1D__MobileNetV2':'B: CNN1D / MobileNetV2',
    'probabilities64':'64 probabilidades','anomaly32':'32 probabilidades de anomalia',
    'mc_mean32':'MC: 32 médias','mc_meanstd64':'MC: 32 médias + 32 std'}

def label(name):
    if name in LABELS:return LABELS[name]
    return name.replace('Autoencoder_CNN_GradCAM','AE+CNN').replace('CNN_BiLSTM','CNN+BiLSTM').replace('IC_','IC: ').replace('__PH_',' / PH: ').replace('icentia11k__','Icentia: ').replace('physionet2017__','PhysioNet: ')

def grid_rows():
    report=json.loads((GRID/'validation_report.json').read_text());assert report['status']=='complete'
    result=[]
    for exp in range(1,6):
        for meta in sorted((GRID/f'exp{exp}').glob('*/*/run_config.json')):
            cfg=json.loads(meta.read_text())
            if cfg.get('status')!='complete' or cfg.get('signature')!=report['signature']:continue
            frame=pd.read_json(meta.with_name('metrics.json'))
            frame=frame[(frame.split=='train')&frame.approach.isin(['local_teacher','global_student'])]
            for approach,g in frame.groupby('approach'):
                if approach=='local_teacher':assert (g.client==g.evaluation_client).all()
                assert not g.evaluation_client.duplicated().any()
                ok=np.isfinite(g.pr_auc_ap);assert ok.any()
                result.append(dict(exp=exp,configuration=cfg['name'],seed=cfg['seed'],role='local' if approach=='local_teacher' else 'global',
                    ap=float(np.average(g.loc[ok,'pr_auc_ap'],weights=g.loc[ok,'fit_examples'])),
                    fit_examples=int(g.fit_examples.sum()),coverage=float(g.loc[ok,'fit_examples'].sum()/g.fit_examples.sum()),
                    origin=','.join(sorted(g.dataset.unique())),source=str(meta.relative_to(ROOT))))
    return result

def final_rows():
    inputs=BASE/'inputs';manifest=json.loads((inputs/'manifest.json').read_text())
    for name,info in manifest['files'].items():assert hashlib.sha256((inputs/name).read_bytes()).hexdigest()==info['sha256']
    validation=json.loads((inputs/'exp6_validation.json').read_text());assert validation['status']=='complete' and validation['validated_pairs']==10
    f=pd.read_json(inputs/'exp6_per_seed.json');f=f[(f.split=='train')&f.approach.isin(['local_teacher','global_student'])]
    rows=[]
    for (candidate,seed,approach),g in f.groupby(['candidate','seed','approach']):
        assert len(g)==2
        rows.append(dict(exp=6,configuration=candidate,seed=seed,role='local' if approach=='local_teacher' else 'global',
            ap=float(np.average(g.pr_auc_ap,weights=g.fit_examples)),fit_examples=int(g.fit_examples.sum()),coverage=1.,
            origin='icentia11k,physionet2017',source='experiment_overview/inputs/exp6_per_seed.json'))
    return rows

def moment_rows():
    from support.softlabels_moments import MODES,complete
    p=MOMENTS/'results/per_seed.csv'
    if not p.exists() or p.stat().st_size<2:return []
    f=pd.read_csv(p);rows=[]
    for r in f.to_dict('records'):
        if r['mode'] not in MODES or not complete(r['mode'],int(r['seed'])):continue
        rows.append(dict(exp=7,configuration=r['mode'],seed=int(r['seed']),role=r['role'],ap=r['pr_auc_ap'],
            fit_examples=2816,coverage=r['fit_coverage'],origin='icentia11k,physionet2017',source='moments_comparison/results/per_seed.csv'))
    return rows

def data():
    frame=pd.DataFrame(grid_rows()+final_rows()+moment_rows())
    assert not frame.duplicated(['exp','configuration','seed','role']).any()
    assert frame.ap.between(0,1).all() and frame.coverage.between(0,1).all()
    result=BASE/'results';result.mkdir(parents=True,exist_ok=True)
    frame.to_csv(result/'all_per_seed.csv',index=False)
    return frame

def paired_stats(frame):
    stats=frame.groupby(['exp','configuration','role']).agg(mean=('ap','mean'),std=('ap','std'),seeds=('seed','nunique'),coverage=('coverage','min')).reset_index()
    local=stats[stats.role=='local'].rename(columns={'mean':'local_ap','std':'local_std','seeds':'n','coverage':'local_coverage'}).drop(columns='role')
    glob=stats[stats.role=='global'].rename(columns={'mean':'global_ap','std':'global_std','seeds':'global_n','coverage':'global_coverage'}).drop(columns='role')
    table=local.merge(glob,on=['exp','configuration'],validate='one_to_one');assert (table.n==table.global_n).all()
    return table.sort_values(['local_ap','configuration'],ascending=[False,True])

def choose(group):
    ranking=paired_stats(group)
    return ranking.iloc[0]

def selections(frame):
    points=[];selected=[]
    for exp in range(1,8):
        f=frame[frame.exp==exp]
        if f.empty:continue
        partial=False;note=''
        if exp==1:
            pieces=[];names=[]
            for origin in ['icentia11k','physionet2017']:
                part=f[f.origin==origin];winner=choose(part).configuration
                pieces.append(part[part.configuration==winner]);names.append(winner)
            merged=pd.concat(pieces);rows=[]
            for (seed,role),g in merged.groupby(['seed','role']):
                assert len(g)==2
                rows.append(dict(exp=exp,configuration=' + '.join(names),seed=seed,role=role,
                    ap=np.average(g.ap,weights=g.fit_examples),coverage=g.coverage.min()))
            selected_frame=pd.DataFrame(rows);note='Média das duas federações separadas; dois modelos globais, não um global conjunto.'
        else:
            if exp==7:
                from support.softlabels_moments import MODES,DESIGN
                available=[set(f.loc[(f.configuration==mode)&(f.role=='local'),'seed']) for mode in MODES]
                common=set.intersection(*available)
                if not common:continue
                f=f[f.seed.isin(common)];partial=common!=set(DESIGN['seeds'])
                note='Variantes comparadas só nas seeds concluídas em comum; parcial.' if partial else 'Cinco seeds completas em todas as variantes.'
            winner=choose(f).configuration;selected_frame=f[f.configuration==winner]
        selected.extend(selected_frame.to_dict('records'))
        stats=choose(selected_frame)
        points.append(dict(exp=exp,configuration=stats.configuration,local_ap=stats.local_ap,global_ap=stats.global_ap,
            local_std=stats.local_std,global_std=stats.global_std,seeds=int(stats.n),partial=partial,
            sample='ampliada' if exp in [4,6,7] else 'smoke',local_epochs=10 if exp>=6 else 1,note=note))
    result=pd.DataFrame(points);selected=pd.DataFrame(selected)
    result.to_csv(BASE/'results/selected_by_experiment.csv',index=False)
    selected.to_csv(BASE/'results/selected_per_seed.csv',index=False)
    return result

def render_experiment(exp):
    from IPython.display import display,Markdown
    display(Markdown(DESCRIPTIONS[exp]+'\n\nPR-AUC local: modelo do cliente no seu próprio holdout. Global: modelo global nos mesmos holdouts. Pesos proporcionais ao fit; não se juntam previsões numa curva única.'))
    if exp==7:
        from support.softlabels_moments_report import render
        render();return
    f=data();part=f[f.exp==exp];table=paired_stats(part)
    table['Configuração']=table.configuration.map(label)
    display(table[['Configuração','local_ap','local_std','global_ap','global_std','n','local_coverage']].rename(columns={
        'local_ap':'PR-AUC local','local_std':'Desvio local','global_ap':'PR-AUC global','global_std':'Desvio global','n':'Seeds','local_coverage':'Cobertura AP'}).round(4))
    if exp==6:
        d=json.loads((BASE/'inputs/exp6_decision.json').read_text())
        display(Markdown('**Escolha da Exp. 6: '+label(d['winner'])+'.** Critério: maior AP local ponderada pelo fit nas cinco seeds. Os resultados históricos com pesos 50/50 não sustentam esta escolha.'))
    if exp<=5:display(Markdown('Três rondas; uma época local por ronda. '+('Na Exp. 4, a seleção interna dos candidatos usa três épocas e a avaliação final usa fit+seleção.' if exp==4 else 'Amostra smoke fixa.')))

def overview_figure(points):
    fig,ax=plt.subplots(figsize=(13,6.5),layout='constrained')
    for exp in [4,6,7]:ax.axvspan(exp-.42,exp+.42,color='#ece8f5',alpha=.7,zorder=0)
    for role,name,color,marker,offset in [('local','Modelos locais — principal','#c3832d','o',-.07),('global','Modelo global','#2d79a8','s',.07)]:
        y=points[f'{role}_ap'];err=points[f'{role}_std'].fillna(0)
        ax.errorbar(points.exp+offset,y,yerr=err,fmt=marker+'-',color=color,capsize=5,label=name,lw=1.5,ms=7)
        for r in points.itertuples():
            value=getattr(r,role+'_ap');sigma=getattr(r,role+'_std');sigma=0 if pd.isna(sigma) else sigma
            ax.annotate(f'{value:.4f}',(r.exp+offset,value),xytext=(0,13 if role=='local' else -19),textcoords='offset points',ha='center',color=color,fontsize=10)
    ax.set_xticks(range(1,8),[f'Exp. {i}'+ ('\nparcial' if i==7 and (points[points.exp==7].empty or points.loc[points.exp==7,'partial'].any()) else '') for i in range(1,8)])
    ax.set_xlabel('Experiência');ax.set_ylabel('PR-AUC no holdout do próprio cliente, ponderado pelo fit')
    ax.set_ylim(0,1.06);ax.set_xlim(.6,7.4);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower left')
    ax.set_title('Comparação exploratória das experiências de Soft Labels\nMesma configuração para as séries local e global em cada experiência',fontweight='bold')
    fig.supxlabel('Barras: desvio entre seeds (Exp. 1–2: uma seed, sem estimativa de dispersão). Fundo lilás: amostra ampliada.\nExp. 1 combina referências separadas; Exp. 6–7 usam dez épocas locais, restantes uma. Linhas apenas guiam a leitura.',fontsize=9)
    return fig

def render_comparisons():
    from IPython.display import display,Markdown
    frame=data();points=selections(frame)
    display(Markdown('A configuração de cada ponto é escolhida pelo **maior PR-AUC local ponderado, em média entre seeds**. O PR-AUC global apresentado pertence à mesma configuração. '
        'A escolha é exploratória e pode beneficiar da seleção entre várias alternativas. Na Exp. 1, escolhe-se uma arquitetura por origem e combina-se a AP por pesos de fit; são duas federações separadas. '
        'Na Exp. 7 usam-se apenas as seeds já concluídas nas quatro variantes, evitando escolher uma variante a partir de seeds diferentes.'))
    fig=overview_figure(points);display(fig);fig.savefig(BASE/'results/pr_auc_by_experiment.png',dpi=170,bbox_inches='tight');plt.close(fig)
    table=points.copy();table.configuration=table.configuration.map(label)
    display(table[['exp','configuration','local_ap','global_ap','seeds','sample','local_epochs','partial']].rename(columns={
        'exp':'Experiência','configuration':'Configuração selecionada','local_ap':'PR-AUC local','global_ap':'PR-AUC global','seeds':'Seeds','sample':'Amostra','local_epochs':'Épocas locais/ronda','partial':'Parcial'}).round(4))
    best=points.loc[points.local_ap.idxmax()]
    display(Markdown(f'**Maior média local observada neste gráfico: Exp. {int(best.exp)}, {label(best.configuration)} ({best.local_ap:.4f}).** '
        'É uma descrição dos resultados disponíveis, não uma prova de superioridade: as amostras, épocas, seeds e número de configurações exploradas diferem. '
        'A Exp. 5 reutiliza o espaço de configurações da Exp. 3. Para escolher a representação das mensagens, a comparação controlada relevante é entre as quatro variantes da Exp. 7, nos mesmos dados e seeds. '
        'A escolha deve considerar também comunicação e cálculo; test continua reservado.'))
    if 7 not in points.exp.values:display(Markdown('**Exp. 7 ainda sem seeds completas comuns às quatro variantes; ponto omitido para não misturar amostras de seeds.**'))
    exp7=frame[frame.exp==7]
    if not exp7.empty and exp7.groupby(['configuration','role']).seed.nunique().eq(5).all() and exp7.configuration.nunique()==4:
        means=exp7.groupby(['configuration','role']).ap.mean().unstack('role')
        current=means.loc['probabilities64'];compact=means.loc['anomaly32']
        display(Markdown('### Combinação recomendada\n\n'
            '**Para maximizar o PR-AUC local, recomenda-se a configuração A: AE+CNN nos clientes Icentia11k, '
            'MobileNetV2 nos clientes PhysioNet 2017 e CNN1D como modelo global, mantendo as probabilidades determinísticas por âncora.** '
            'A configuração A foi selecionada na Exp. 6 pelo PR-AUC local ponderado nas cinco seeds. '
            f'Na Exp. 7, a representação atual de 64 probabilidades teve a maior média local ({current["local"]:.4f}), '
            f'com PR-AUC global de {current["global"]:.4f}. As variantes MC Dropout, com ou sem std, '
            'não melhoraram a média local nesta experiência e acrescentaram custo de cálculo; não são a primeira opção para prosseguir.\n\n'
            '**Para wearables com prioridade à comunicação, a alternativa é a mesma configuração A enviando só 32 probabilidades de anomalia**, '
            f'com PR-AUC local de {compact["local"]:.4f} e global de {compact["global"]:.4f}. '
            f'A diferença local observada face aos 64 valores é {current["local"]-compact["local"]:.4f} '
            f'({100*(current["local"]-compact["local"]):.2f} pontos percentuais). '
            'Esta alternativa reduz o payload de probabilidades em 50% e permite reconstruir a probabilidade normal por `1 − p(anomalia)`. '
            'O download do modelo global e as mensagens adicionais de um eventual protocolo SA não são reduzidos nessa proporção. '
            'A equivalência é matemática; as pequenas diferenças medidas podem resultar de arredondamentos e da sua propagação entre rondas, '
            'não demonstrando superioridade estatística de uma codificação.\n\n'
            '**Decisão:** manter A e a distilação determinística; usar 64 valores quando a prioridade estrita é a maior média local observada, '
            'ou 32 valores quando se privilegia o compromisso com a comunicação em wearables. '
            'Esta escolha é exploratória e não demonstra maior privacidade, menor consumo energético ou desempenho em novos pacientes. '
            'A validade das âncoras sintéticas e a generalização do modelo global continuam a exigir avaliação; o conjunto de teste permanece reservado.'))

def capture_outputs(function,*args):
    import IPython.display
    outputs=[]
    def capture(value,**kwargs):
        if hasattr(value,'savefig'):
            buffer=io.BytesIO();value.savefig(buffer,format='png',dpi=130,bbox_inches='tight')
            result={'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':'PR-AUC local e global por experiência'}
        elif hasattr(value,'_repr_markdown_'):result={'text/markdown':value._repr_markdown_(),'text/plain':str(value)}
        elif hasattr(value,'to_html'):result={'text/html':value.to_html(index=False),'text/plain':repr(value)}
        else:result={'text/plain':repr(value)}
        outputs.append(nbformat.v4.new_output('display_data',data=result))
    old=IPython.display.display;IPython.display.display=capture
    try:function(*args)
    finally:IPython.display.display=old
    return outputs

def refresh():
    BASE.mkdir(parents=True,exist_ok=True)
    setup="from pathlib import Path\nimport sys\nroot=Path.cwd().resolve()\nwhile not (root/'requirements.txt').exists() and root!=root.parent: root=root.parent\nsys.path.insert(0,str(root))"
    cells=[nbformat.v4.new_markdown_cell('# Experiências de Soft Labels — Exp. 1 a Exp. 7\n\nResultados da versão ponderada pelo fit. Exp. 6 corresponde à confirmação A/B; Exp. 7, à comparação das mensagens com MC Dropout. '
        'Os modelos locais e o modelo global são avaliados separadamente nos holdouts dos próprios clientes. Os resultados são exploratórios; test reservado.'),nbformat.v4.new_code_cell(setup)]
    for exp in range(1,8):
        cells.append(nbformat.v4.new_markdown_cell(f'## Exp. {exp} — {TITLES[exp]}'))
        cells.append(nbformat.v4.new_code_cell(f'from support.softlabels_overview import render_experiment\nrender_experiment({exp})',execution_count=exp,outputs=capture_outputs(render_experiment,exp)))
    cells.append(nbformat.v4.new_markdown_cell('## Comparisons — comparação entre experiências'))
    cells.append(nbformat.v4.new_code_cell('from support.softlabels_overview import render_comparisons\nrender_comparisons()',execution_count=8,outputs=capture_outputs(render_comparisons)))
    notebook=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'ordered_softlabels_experiments':1})
    nbformat.validate(notebook);temp=PATH.with_suffix('.tmp.ipynb');nbformat.write(notebook,temp);temp.replace(PATH)
    return selections(data())

def watch():
    """Keep the already-running older reporter from putting Exp. 7 above Exp. 1."""
    BASE.mkdir(parents=True,exist_ok=True)
    with (BASE/'watch.lock').open('a+') as watcher_lock:
        fcntl.flock(watcher_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        previous=None
        while True:
            state=tuple((str(p),p.stat().st_mtime_ns) for p in sorted(MOMENTS.glob('*/artifacts/*/validation.json')))
            try:
                notebook=nbformat.read(PATH,4)
                legacy=any('softlabels_moments_comparison' in c.metadata.get('tags',[]) for c in notebook.cells)
                if state!=previous or legacy:
                    refresh();previous=state;print('ORDERED NOTEBOOK UPDATED',len(state),'new validated runs',flush=True)
                with (MOMENTS/'run.lock').open('a+') as training_lock:
                    try:
                        fcntl.flock(training_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        fcntl.flock(training_lock,fcntl.LOCK_UN);refresh();print('FINAL NOTEBOOK UPDATED',flush=True);return
                    except BlockingIOError:pass
            except (OSError,ValueError) as error:print('Waiting for next report update:',error,flush=True)
            time.sleep(15)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    watch() if args.watch else refresh()
