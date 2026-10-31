# Deteção de arritmias — protocolo ativo com modelo global

O fluxo anterior de soft labels sem aluno global foi retirado. A recomendação de RF foi retirada. O novo servidor treina uma CNN1D global e redistribui os seus pesos; FedAvg agrega e redistribui parâmetros compatíveis. RF/KNN e as duas GRU não participam no FL ativo.

```sh
.venv/bin/python run_global_experiments.py --pilot
.venv/bin/python run_global_experiments.py
```

Detalhes: [FedAvg](<FedAvg tests/README.md>), [soft labels](<SoftLabels tests/README.md>) e [explicação das experiências](<SoftLabels tests/explicacao_experiencias.txt>).

- `Tests/`: apenas os três notebooks centralizados; apoio em `support/model_tests/`.
- `support/federated_global/`: implementação ativa, validação, relatórios e notebooks.
- `artifacts/history_before_global_student/`: código/notebooks históricos e manifesto; resultados anteriores preservados nas respetivas pastas de artifacts.
- `Privacy Techniques/`: preservada, incluindo quando vazia.

Os dados e outputs preparados permanecem locais e excluídos de Git. O test é reservado. Histórico de obtenção/preparação e configuração do ambiente no README arquivado em `artifacts/history_before_global_student/README_original.md`.


Estudo de privacidade local preparado em [Privacy Techniques/LDP](<Privacy Techniques/LDP/README.md>): logits perturbados no cliente, composição por ciclo e curvas de utilidade. Aguarda a candidata final antes de treino real.
