"""Reproduce the four-condition summary from completed runs; no training."""
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/softlabels-comparison-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/softlabels_overview/comparacao_privacidade'
SOURCES = {
    'Sem SA / sem LDP': 'Privacy Techniques/SA/Communication/Softlabels/artifacts/{seed}/plain',
    'Só SA': 'Privacy Techniques/SA/Communication/Softlabels/artifacts/{seed}/SA',
    'Só LDP (P2)': 'artifacts/ldp_logits/closed_loop/runs/P2_ldp/artifacts/{seed}',
    'SA + LDP (P2)': 'artifacts/ldp_sa/runs/artifacts/{seed}',
}
METRICS = {'pr_auc_ap': 'PR-AUC (AP) ↑', 'recall': 'Sensibilidade ↑', 'far': 'Taxa de falsos alarmes ↓'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, provenance = [], []
    registries = {}
    for condition, template in SOURCES.items():
        for seed in range(2031, 2036):
            folder = ROOT / template.format(seed=seed)
            meta = json.loads((folder / 'run.json').read_text())
            assert meta['status'] == 'complete' and meta['seed'] == seed
            assert meta['settings']['rounds'] == 3 and meta['settings']['local_epochs'] == 10
            registry = meta['initial_data_registry']
            if seed in registries:
                assert registry == registries[seed], 'Different samples between conditions'
            registries[seed] = registry
            frame = pd.read_json(folder / 'metrics.json')
            for role in ['local', 'global']:
                q = frame[(frame['round'] == 3) & (frame.role == role) & (frame.split == 'holdout')]
                assert len(q) == 22 and not q.client.duplicated().any()
                assert q.fit_examples.sum() == 2816
                rows.append(dict(condition=condition, seed=seed, role=role,
                                 **{m: np.average(q[m], weights=q.fit_examples) for m in METRICS}))
            provenance.append(dict(condition=condition, seed=seed, source=str(folder.relative_to(ROOT)), signature=meta['signature']))
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT / 'por_seed.csv', index=False)
    summary = per_seed.groupby(['condition', 'role'], sort=False)[list(METRICS)].agg(['mean', 'std'])
    summary.to_csv(OUT / 'resumo.csv')
    (OUT / 'fontes.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2))
    colors = ['#49677f', '#179d91', '#d79531', '#9269ad']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharey=True)
    for r, role in enumerate(['local', 'global']):
        for ax, (metric, title) in zip(axes[r], METRICS.items()):
            for i, condition in enumerate(SOURCES):
                values = per_seed[(per_seed.condition == condition) & (per_seed.role == role)][metric].to_numpy()
                mean, std = values.mean(), values.std(ddof=1)
                ax.bar(i, mean, color=colors[i], width=.64, alpha=.85)
                ax.errorbar(i, mean, yerr=std, fmt='none', color='#25303a', capsize=4, lw=1.3)
                ax.scatter(i + np.linspace(-.17, .17, 5), values, color='#25303a', s=14, zorder=3, alpha=.7)
                ax.text(i, 1.075, f'{mean:.3f} ± {std:.3f}', ha='center', fontsize=9)
            ax.set_title(title, pad=33, fontweight='bold')
            ax.set_xticks(range(4), ['Sem SA\nsem LDP', 'Só SA', 'Só LDP', 'SA + LDP'])
            ax.set_ylim(0, 1.02)
            ax.set_yticks(np.arange(0, 1.01, .2))
            ax.grid(axis='y', alpha=.18)
            ax.set_axisbelow(True)
        axes[r, 0].set_ylabel(('Modelos locais' if role == 'local' else 'Modelo global') + '\nMédia ponderada por cliente')
    fig.suptitle('Soft labels — comparação das quatro configurações', fontsize=20, fontweight='bold', y=.985)
    fig.text(.5, .925, 'Ronda 3 · 5 seeds (2031–2035) · 22 clientes · mesmos holdouts · pesos pelo número de exemplos de fit', ha='center', color='#425466')
    fig.text(.06, .065, 'Barras: média entre seeds; erros: ±1 desvio padrão; pontos: seeds individuais. AP é média por cliente, não AP pooled.\nLDP P2: Gaussiano, margem, C=0,5; ε=4 e δ=10⁻⁵ por cliente/ciclo. SA: 24 bits, limiar 12/22.\nRuído LDP independente entre execuções: a diferença Só LDP ↔ SA + LDP não isola o efeito da SA. Estes resultados são de holdout, não de test.', fontsize=10, linespacing=1.6)
    fig.subplots_adjust(top=.82, bottom=.19, left=.07, right=.985, hspace=.61, wspace=.15)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'comparacao_softlabels_privacidade.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    (OUT / 'LEITURA.md').write_text('# Comparação de soft labels\n\nResultados da ronda final (3), mesmos 22 clientes e cinco seeds. Cada métrica é primeiro ponderada pelo número de exemplos de fit por cliente e depois resumida pela média e desvio padrão amostral entre seeds. Modelos locais e global são avaliados nos mesmos holdouts dos clientes de treino. Não se misturam estes valores com o test final de LDP + SA.\n\nLDP usa P2 (Gaussiano, margem, C=0,5, ε=4 e δ=10⁻⁵ por cliente/ciclo); SA usa 24 bits e limiar 12/22. O ruído LDP independente impede atribuir diferenças entre as duas condições privadas apenas à SA.\n\nFontes e assinaturas em `fontes.json`; valores por seed em `por_seed.csv`; médias e desvios em `resumo.csv`. Regenerar com `.venv/bin/python support/softlabels_privacy_comparison.py`.\n', encoding='utf-8')
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
