# Protocolo ativo — professores heterogéneos e aluno global CNN1D

Fluxo aprovado: professores locais -> logits nas âncoras -> aluno CNN1D treinado no servidor -> pesos do aluno distribuídos a todos os clientes -> orientação dos professores na ronda seguinte. O aluno é um modelo treinado, não um ensemble de previsões de avaliação.

A CNN1D global tem um trunk e classificador partilhados e pequenos adaptadores por origem para as 17/19 features nativas. Aceita os comprimentos de sinal nativos com pooling mascarado. A mesma estrutura completa e os mesmos pesos são distribuídos a todos os clientes; o cliente seleciona o adaptador da sua origem. Professores podem ter arquiteturas diferentes e recebem o aluno global completo, usando as suas previsões nas âncoras para distilação local. Não se tenta carregar diretamente pesos CNN1D numa arquitetura diferente.

Os professores treinam uma época supervisionada por ronda; nas rondas 2/3 fazem também um passo de distilação com o aluno recebido. Enviam só logits de 32 âncoras sintéticas. O servidor converte logits para probabilidades a temperatura 1, calcula o alvo suave (metade do peso por origem; igual por cliente dentro da origem) e treina o aluno durante três passos/épocas conjuntas por ronda, sem exemplos privados ou labels clínicas. Há três rondas; o aluno final é redistribuído e guardado em todos os clientes. A avaliação é offline e não entra nas mensagens nem no treino do servidor.

RF e KNN excluídos: a adaptação anterior por ensemble/consenso não corresponde ao modelo global redistribuído solicitado. GRU excluídas por custo. A escolha anterior de RF está retirada até concluir e analisar as novas experiências.

A grelha é gerada APÓS completar o novo FedAvg, usando os top 3 por origem do modelo global entregue nos holdouts locais, média de três seeds. Não se reutiliza o ranking RF/KNN anterior:

- exp1: cada família da união dos top 3, federações separadas por origem, seed 2026.
- exp2: top 3 Icentia x top 3 PhysioNet, seed 2026.
- exp3: todas as atribuições da união x união, três seeds, controlos privados.
- exp4: mesma amostra agrupada da exp4 histórica; seleção comum/por origem/por cliente entre a nova união; cinco seeds, 15 pares. Normalização por origem ajustada ao fit, em coordenadas comuns ao aluno; refit inclui seleção, nunca holdout/dev. Esta normalização difere da antiga normalização privada por cliente e deve ser declarada.
- exp5: nove cruzamentos top 3 x top 3 e os controlos homogéneos adicionais da união, três seeds.

Nas exp1/2/3/5, mantêm-se IDs e exemplos smoke do FedAvg. A exp4 mantém a sua amostra ampliada com fit/seleção/holdout separados por registo. Nunca se usa test. A representação nativa é mantida; âncoras de batimento/registo são vistas aproximadas do mesmo latente e não têm validação clínica.

Todos os pares incluem controlo exclusivamente privado a partir da mesma inicialização: três épocas privadas e duas épocas privadas adicionais em lugar dos dois passos de KD pública. Isto não iguala exemplos processados, tempo ou FLOPs. A seleção da exp4 usa modelos de três épocas privadas na validação interna, não os resultados do holdout/dev. Na nova exp4 não se calibra um limiar tuned; métricas usam argmax/0,5, dev/holdout não ajustam limiares. Ajuste de limiar exige validação interna própria antes do test final.

Executar tudo: `.venv/bin/python run_global_experiments.py`. Só soft labels após o novo FedAvg: `.venv/bin/python "SoftLabels tests/run_all.py"`. Selecionar uma experiência: acrescentar `--experiment exp4`. Piloto: `--pilot`.

Resultados: `artifacts/global_student_v1/`. Notebooks ativos: `comparation_softlabels.ipynb` e `FedAvg tests/comparation_fedavg.ipynb`. Validar: `.venv/bin/python run_global_experiments.py --phase validate`. Regenerar relatórios: `--phase report`. Testes pequenos: `.venv/bin/python support/model_tests/verify_global.py`.

O código e os notebooks do consenso sem aluno global foram retirados da estrutura ativa e arquivados na raiz em `artifacts/history_before_global_student/`. Os resultados científicos históricos foram preservados. Não devem ser usados para concluir que o novo aluno global funciona nem para fixar RF como modelo FL.

## Comparação final das duas candidatas

Em `final candidates/`: A (AE+CNN/MobileNetV2) vs B (CNN1D/MobileNetV2), aluno CNN1D em ambas. Cinco seeds novas 2031–2035, dez épocas locais por ronda (30 totais), mesma amostra ampliada agrupada e controlos. Critério de escolha congelado antes do treino; resultados e decisão no notebook `final candidates/resultados.ipynb`. Test reservado.
