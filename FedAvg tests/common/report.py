"""Relatório ativo exclusivamente binário; não mistura experiências históricas."""
import json
import pandas as pd
from .data import ROOT,DATA_ROOT
from .learning import DISPLAY_NAMES
METRICS=['accuracy','f1','precision','recall','roc_auc','pr_auc_ap','miss_rate','far','auc_valid_classes']

def build_report(args):
    frames=[]
    datasets=['physionet2017','icentia11k'] if args.dataset=='both' else [args.dataset]
    for ds in datasets:
        root=DATA_ROOT/f'{ds}_{args.train_cap}_{args.eval_cap}_{args.seed}'
        for config in sorted(root.glob('*/run_config.json')):
            cfg=json.loads(config.read_text())
            if cfg['status']!='complete':continue
            assert cfg['task']=='binary' and cfg['protocol_id']=='binary_smoke_v3' and not cfg['test_used']
            df=pd.read_json(config.parent/'metrics.json');frames.append(df[df['round']==args.rounds])
    if not frames:return
    final=pd.concat(frames,ignore_index=True);final['model']=final.model.replace(DISPLAY_NAMES)
    selected=final[(final.split=='train')&final.approach.isin(['local_only','local_after_fedavg_training','global'])]
    summary=selected.groupby(['dataset','model','approach'])[METRICS].mean().reset_index()
    counts=selected.groupby(['dataset','model','approach']).agg(clients=('evaluation_client','size'),valid_pr_clients=('pr_auc_ap','count')).reset_index()
    summary=summary.merge(counts,on=['dataset','model','approach']).sort_values(['dataset','pr_auc_ap','roc_auc'],ascending=[True,False,False])
    summary.to_json(DATA_ROOT/'local_comparison.json',orient='records',double_precision=15)
    final.to_json(DATA_ROOT/'all_final_metrics.json',orient='records',double_precision=15)
    lines=['FLOWER / FEDAVG — DETEÇÃO BINÁRIA DE ANOMALIAS','',
        'Todas as aprendizagens são binárias: N=normal (0), A/O no PhysioNet ou S/V no Icentia=anomalia (1). ~ e Q excluídos antes de treinar, normalizar e avaliar.',
        'Protocolo binary_smoke_v3: mesmos IDs da seleção shared_smoke_v2, filtrados. PhysioNet 354 fit / 90 holdout / 59 dev; Icentia 792 fit / 203 holdout / 94 dev. Test não é carregado.',
        'Ranking principal: PR-AUC da classe anomalia (average precision), ROC-AUC positivo como desempate. F1, precision, recall, miss rate e FAR referem-se à anomalia.',
        'Miss rate=FN/(TP+FN)=1−recall; FAR=FP/(FP+TN), não alarmes/hora. Decisão argmax; empate prevê normal.',
        'Local e global no mesmo holdout, média não ponderada entre clientes. AP/AUC indefinidas por ausência de classes são omitidas com cobertura reportada.',
        'Doze modelos usam Flower FedAvg ponderado por fit elegíveis. KNN/RF usam soft voting, não FedAvg. A agregação das suas previsões não atualiza os modelos locais.',
        'local_only: treino isolado; local_after_fedavg_training: modelo do cliente após treino na última ronda, antes da agregação; global: modelo agregado no mesmo holdout.',
        '', 'MÉTRICAS LOCAIS — PRIORIDADE',summary.round(4).to_string(index=False)]
    for ds in datasets:
        part=summary[(summary.dataset==ds)&(summary.approach=='local_after_fedavg_training')].sort_values(['pr_auc_ap','roc_auc'],ascending=False)
        if len(part):
            best=part.iloc[0];lines+=['',f"{ds}: candidato local FedAvg {best['model']}; AP da anomalia={best['pr_auc_ap']:.4f}."]
    lines+=['','MÉTRICAS POR CLIENTE',selected[['dataset','model','approach','evaluation_client']+METRICS].round(4).to_string(index=False),
        '', 'GENERALIZAÇÃO GLOBAL EM DEV (POOLED; FORA DO RANKING LOCAL)',final[(final.approach=='global_pooled')&(final.split=='dev')][['dataset','model']+METRICS].round(4).to_string(index=False),
        '', 'GENERALIZAÇÃO DOS MODELOS LOCAIS EM DEV',final[(final.split=='dev')&final.approach.isin(['local_only_external','local_federated_external'])].groupby(['dataset','model','approach'])[METRICS].mean().round(4).to_string(),
        '', 'LIMITAÇÕES',
        'A seleção de origem foi enriquecida por classes clínicas; o treino usa apenas a sua redução binária. Prevalência e pequenas amostras afetam AP.',
        'Holdout Icentia partilha o paciente com fit; dev mantém pacientes separados. Não é avaliação clínica final.',
        'SVM central é RBF e FL linear. Log Reg central usa sklearn e FL AdamW. As redes mantêm arquiteturas reduzidas e três épocas totais; otimizadores FL reiniciam por ronda.',
        'Estatísticas de normalização usam apenas fit binários, e são partilhadas. Não é privacidade diferencial, agregação segura ou inferência privada.',
        'Resultados históricos não entram nesta análise. Artefactos novos em FedAvg tests/artifacts/binary_smoke_v3/.']
    target=ROOT/'FedAvg tests/analise_comparativa.txt';target.write_text('\n'.join(lines)+'\n');print('REPORT',target,flush=True)
