# Protocolo ativo — FedAvg com modelo global

Execução: `.venv/bin/python "FedAvg tests/run_all.py"` ou `.venv/bin/python run_global_experiments.py --phase fedavg`.

Dez candidatos com parâmetros compatíveis; RF/KNN não participam (o ensemble anterior não é FedAvg). GRU5 e GRU simplificada excluídos por custo, conforme pedido anterior. Três seeds de treino (2026–2028), dados fixos à seed 2026, três rondas e uma época local por ronda: 60 execuções modelo/dataset/seed. Mesmos exemplos smoke elegíveis: PhysioNet 354 fit/90 holdout/59 dev; Icentia 792/203/94. Test reservado.

Servidor distribui pesos iniciais -> treino local -> agregação ponderada pelo fit -> modelo global atualizado -> redistribuição em todas as rondas. Guardam-se checkpoints globais por ronda e réplicas finais idênticas por cliente, com hashes auditados. Os buffers inteiros de BatchNorm são convertidos ao seu dtype original após a agregação; não são parâmetros treináveis.

Resultados: `artifacts/global_model_v1/`. Comparativo: `comparation_fedavg.ipynb`. Ranking principal para a nova seleção: AP do modelo global entregue aos clientes nos holdouts locais, média entre três seeds, ROC-AUC como desempate. Isto é seleção exploratória; o desempenho local antes da agregação aparece separadamente. Não se misturam estes rankings com referências históricas de RF/KNN por soft voting.

Os resultados históricos mantêm-se nas pastas anteriores de artifacts. Scripts/notebooks anteriores estão arquivados em `artifacts/history_before_global_student/` na raiz do projeto. Os módulos common/data.py, models.py, learning.py e metrics.py continuam a ser partilhados com os testes centralizados; a disponibilidade de RF nesses módulos não o torna candidato do FL ativo.
