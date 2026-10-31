"""User-approved analysis amendment: choose A/B by LOCAL teacher AP in dev."""
import fcntl,hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from support import final_candidates as final
from support.federated_global.shared import dump

AMENDMENT=final.FOLDER/'analysis_amendment.json'
METRIC='local_teacher_dev'

def signature():
    return hashlib.sha256(Path(__file__).read_bytes()+AMENDMENT.read_bytes()+final.signature().encode()).hexdigest()


def score(per_seed,timings):
    local=per_seed[(per_seed.approach=='local_teacher')&(per_seed.split=='dev')]
    expected={(candidate,ds,seed) for candidate in final.CANDIDATES for ds in final.DOMAINS for seed in final.SEEDS}
    assert set(local[['candidate','dataset','seed']].itertuples(index=False,name=None))==expected
    assert not local.duplicated(['candidate','dataset','seed']).any() and np.isfinite(local.pr_auc_ap).all()
    mean=local.groupby(['candidate','dataset']).pr_auc_ap.mean().unstack('dataset')
    regret=mean.max(axis=0)-mean
    rank=pd.DataFrame(dict(worst_origin_regret=regret.max(axis=1),mean_regret=regret.mean(axis=1),total_seconds=timings.groupby('candidate').elapsed_seconds.sum()))
    rank=rank.sort_values(['worst_origin_regret','mean_regret','total_seconds'])
    tied=rank[rank.worst_origin_regret<=rank.worst_origin_regret.min()+1e-6]
    tied=tied[tied.mean_regret<=tied.mean_regret.min()+1e-6]
    winner=tied.assign(candidate_name=tied.index).sort_values(['total_seconds','candidate_name']).index[0]
    a,b=list(final.CANDIDATES)
    paired=local.pivot(index=['dataset','seed'],columns='candidate',values='pr_auc_ap');paired['A_minus_B']=paired[a]-paired[b]
    differences=paired.groupby('dataset').A_minus_B.agg(['mean','std','min','max']).reset_index()
    differences['positive_seeds_A']=paired.groupby('dataset').A_minus_B.apply(lambda v:int((v>1e-12).sum())).values
    return winner,mean,rank,paired,differences


def finalize():
    """Run only after the original job releases its lock and validates all ten runs."""
    if not AMENDMENT.exists():return False
    amendment=json.loads(AMENDMENT.read_text())
    assert amendment['training_signature']==final.signature() and amendment['primary_metric']==METRIC
    lock=(final.BASE/'run.lock').open('a+')
    try:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
        report=final.BASE/'validation_report.json'
        if not report.exists():return False
        validation=json.loads(report.read_text())
        if validation.get('status')!='complete' or validation.get('validated_pairs')!=final.EXPECTED_PAIRS:return False
        assert validation['signature']==final.signature()
        frame,timings=final.collect()
        if len(timings)!=final.EXPECTED_PAIRS:return False
        keys=['candidate','dataset','split','approach']
        per=frame.groupby(keys+['seed'])[final.METRICS].mean().reset_index()
        winner,means,rank,paired,differences=score(per,timings)
        history=final.BASE/'analysis_history';history.mkdir(exist_ok=True)
        for source,name in [(final.FOLDER/'decision.json','global_decision_before_user_amendment.json'),
                            (final.FOLDER/'decisao_final.txt','global_reasoning_before_user_amendment.txt'),
                            (final.BASE/'paired_differences.json','global_paired_differences.json'),
                            (final.BASE/'decision_scores.json','global_decision_scores.json')]:
            target=history/name
            if source.exists() and not target.exists():shutil.copy2(source,target)
        choice=dict(status='selected_after_validation',winner=winner,teachers=final.CANDIDATES[winner],student='CNN1D',
            signature=final.signature(),analysis_signature=signature(),analysis_priority=METRIC,
            criterion='Minimize worst-origin LOCAL teacher AP regret in dev; mean regret, runtime and name break ties within 1e-6',
            paired_runs=final.EXPECTED_PAIRS,test_used=False,
            limitation='User changed analysis priority during training; dev already explored. Configuration selection, not independent test.')
        differences.to_json(final.BASE/'paired_differences.json',orient='records',double_precision=15)
        rank.reset_index(names='candidate').to_json(final.BASE/'decision_scores.json',orient='records',double_precision=15)
        dump(final.FOLDER/'decision.json',choice)
        text='''DECISÃO FINAL — PRIORIDADE AOS PROFESSORES LOCAIS EM DEV

Candidata escolhida: '''+winner+'\nProfessores por origem: '+str(final.CANDIDATES[winner])+'''
Aluno global CNN1D: resultado complementar, não o critério de escolha.

Cada cliente tem um professor próprio. A AP local em dev é calculada por
professor/cliente de avaliação, com média igual dentro da origem e depois
média das cinco seeds completas. Não é a AP do aluno nem a de um ensemble.

AP LOCAL EM DEV — cinco seeds:
'''+means.to_string()+'''\n\nPERDAS RELATIVAS AO MELHOR LOCAL DE CADA ORIGEM:
'''+rank.to_string()+'''\n\nDIFERENÇAS PAREADAS LOCAIS A − B:
'''+differences.to_string(index=False)+'''

Critério: minimizar a maior perda de AP dentro de cada origem; desempate
pela perda média, tempo total e nome (tolerância 10^-6). Mantém-se a regra
de compromisso, mudando a métrica de aluno global para professores locais
por pedido explícito do utilizador. O aluno global e os controlos privados
continuam nas tabelas e gráficos como resultados complementares.

Esta alteração da análise foi registada durante o treino, sem reiniciar
modelos nem alterar dados, seeds, épocas ou checkpoints. O desenho de treino
original e a decisão que usava o global ficam preservados no histórico da
análise. Dev já foi consultado: a escolha é exploratória e não um teste
independente. O test continua reservado. Não se escolhe a melhor seed nem
a melhor ronda com base em dev.
'''
        (final.FOLDER/'decisao_final.txt').write_text(text)
        status=json.loads((final.BASE/'status.json').read_text())
        status.update(winner=winner,decision_metric=METRIC,analysis_signature=signature())
        dump(final.BASE/'status.json',status)
        return True
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
