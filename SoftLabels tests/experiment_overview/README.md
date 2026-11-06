# Organização e comparação das Exp. 1–7

`../comparation_softlabels.ipynb` está organizado por experiências e termina com **Comparisons**. As tabelas mostram todas as configurações locais/globais; o gráfico final escolhe uma configuração por experiência pelo maior AP local médio ponderado pelo fit e mostra o global da mesma configuração. O AP é calculado no holdout do próprio cliente e não juntando previsões numa curva única.

Exp. 1–5: versão ponderada validada em `../artifacts/global_student_sample_weighted_v2/`, usando os resultados individuais e os tamanhos de fit efetivos. Exp. 6: snapshot de `testing-models` (commit e SHA-256 em `inputs/manifest.json`), pois os artefactos finais tradicionais nesta branch ainda incluem resultados históricos 50/50. A nova versão seleciona A, pelo AP local ponderado, e não B pelo critério antigo. Exp. 7: condições completas e validadas em `../moments_comparison/`.

Exp. 1 usa duas federações separadas: seleciona-se a melhor arquitetura local por origem e combina-se a AP com os pesos de fit. O ponto global representa a média dos dois modelos globais separados, não uma federação conjunta. Exp. 7 usa exclusivamente as seeds completas comuns às quatro variantes; até ter cinco seeds, está marcada como parcial.

Amostras e orçamentos diferem: Exp. 1–3/5 usam a amostra smoke; Exp. 4/6/7 usam a ampliada; Exp. 6/7 têm dez épocas locais por ronda, restantes uma. A grelha Exp. 5 é subconjunto de Exp. 3. As escolhas entre configurações são exploratórias e usam holdouts, pelo que o máximo no gráfico não constitui prova de superioridade nem teste independente. Test continua reservado.

Gráfico: `results/pr_auc_by_experiment.png`; pontos: `selected_by_experiment.csv`; valores por seed e configuração em `all_per_seed.csv` e `selected_per_seed.csv`.

Regenerar: `.venv/bin/python -m support.softlabels_overview`. O relatório da Exp. 7 chama esta organização automaticamente em novas execuções. Enquanto a execução já iniciada mantiver o gerador anterior em memória, `--watch` repõe a organização e atualiza o gráfico após novas condições completas; não altera o treino. Backup anterior: `artifacts/notebook_history/comparation_softlabels_before_ordered_sections.ipynb`.
