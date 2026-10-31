# Final candidates — comparação A/B por distilação

Aprovadas as duas candidatas, ambas com aluno global CNN1D:

| Candidata | Professor Icentia | Professor PhysioNet |
|---|---|---|
| A | Autoencoder + CNN (Grad-CAM) | MobileNetV2 |
| B | CNN1D | MobileNetV2 |

Não inclui FedAvg, RF, KNN ou GRU. Mantém o fluxo professores → logits nas âncoras → treino CNN1D no servidor → distribuição do aluno global → orientação dos professores na ronda seguinte.

## Comparação congelada

- Cinco seeds novas e pareadas: 2031–2035; dez execuções, sem reutilizar as seeds 2026–2028 como nova evidência.
- Mesma amostra ampliada e agrupada da exp4 para ambas: 128 fit (fit + seleção original) e 32 holdout por cliente train; até 128 dev por cliente. Icentia: 16 train/2 dev; PhysioNet: 6 train/1 dev. Mantém os grupos/IDs, sem abrir test. Não usa todos os dados disponíveis.
- Normalização por origem só no fit+seleção; holdout e dev não entram no ajuste. As cinco seeds mantêm os dados fixos e medem variabilidade de treino, não novos pacientes.
- Três rondas, dez épocas supervisionadas locais por ronda (30 no total), 32 âncoras sintéticas e três passos conjuntos do aluno por ronda, com igual peso por origem. O aluno completo tem trunk/classificador partilhados e adaptadores de features por origem.
- A tem duas épocas privadas adicionais de reconstrução do autoencoder no início, tal como no protocolo anterior. O orçamento de comunicação é igual; tempo/FLOPs e custo de treino não são iguais. Reportar os tempos, sem atribuir tudo apenas à arquitetura.
- Cada execução mantém os controlos privados e valida logits, consenso, checkpoints globais, réplicas dos clientes e métricas das previsões.

## Regra atual para decidir — alteração de análise aprovada

Métrica principal: PR-AUC dos **professores locais em dev**, média igual entre clientes dentro da origem e depois média entre as cinco seeds.

Para cada origem, calcula-se a diferença entre cada candidata e a melhor candidata dessa origem. Escolhe-se a candidata com **menor perda máxima de AP entre as origens** (minimax regret). Assim, compara-se a perda dentro da mesma origem e limita-se o sacrifício de uma origem para favorecer a outra. Empates até 0,000001: menor perda média; depois menor tempo total e nome.

Também são apresentados AP/desvio, ROC-AUC, recall, FAR, F1, diferenças pareadas A−B e número de seeds em que A vence. O critério é descritivo, não um teste de significância. Nenhum checkpoint/seed é escolhido por ser o melhor em dev: decide-se a configuração, com a última ronda fixa.

Dev e estes clientes já foram explorados. Esta é a decisão final entre as duas configurações para o trabalho, não uma avaliação final independente nem prova clínica. O test permanece reservado para confirmar o protocolo congelado. Não ajustar limiares no test.

## Executar na raiz

```sh
.venv/bin/python "SoftLabels tests/final candidates/run.py"
```

O desenho em `design.json` é congelado antes do treino. Um lock impede duplicação. A execução reutiliza somente pares completos com a assinatura correta; uma execução interrompida recomeça o par, não a última ronda. O notebook `resultados.ipynb` é atualizado após cada par, com imagens incorporadas. Os resultados e checkpoints ficam em `artifacts/` nesta pasta; o estado está em `artifacts/status.json` e o log em `artifacts/execution.log`.

Ao completar dez execuções e a validação, cria `decision.json` e `decisao_final.txt`, com a escolha efetiva e a justificação. Não apresenta uma vencedora enquanto faltarem execuções/validação. Não altera as experiências históricas nem as suas assinaturas.

O controlo privado faz 30 épocas supervisionadas e duas épocas privadas adicionais em lugar dos dois passos de KD pública. A CNN1D global é reinicializada por seed e percorre o ciclo inteiro cliente–servidor: as cinco seeds não são apenas repetições da avaliação.

## Alteração da prioridade durante o treino

Por confirmação explícita do utilizador ("Sim, local em dev"), a escolha final usa agora AP dos professores locais em dev. O global é complementar. `analysis_amendment.json` regista esta alteração; o desenho e a assinatura do treino original ficam intactos para não reiniciar as execuções. O mesmo critério de perda máxima relativa por origem é aplicado à AP local. A alteração ocorreu após resultados parciais e a conclusão continua exploratória.

Após as dez execuções e validação, o pós-processamento local atualiza `decision.json`, `decisao_final.txt`, as diferenças pareadas e o notebook. A análise que usava AP global é preservada em `artifacts/analysis_history/`. LDP aguarda a decisão local validada e não pode usar uma escolha global desatualizada.
