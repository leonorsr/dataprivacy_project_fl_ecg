# Resultados preliminares — Local Differential Privacy (LDP)

## 1. Objetivo

Documentar os resultados das experiências com Local Differential Privacy (LDP) aplicadas à partilha de logits no contexto de Federated Learning para deteção de arritmias cardíacas.

Os resultados apresentados são preliminares. A comparação com os baselines sem privacidade ainda não foi concluída.

## 2. Protocolo experimental

| Parâmetro | Configuração |
|---|---|
| Unidade de privacidade | Client replacement |
| Clientes | 22 |
| Rondas por execução | 3 |
| Épocas locais por ronda | 10 |
| Limite de clipping L1 (C) | 32 |
| Mecanismo | Laplace |
| Orçamento por execução (ε) | 0,8 |
| δ | 0 |
| Seeds | 2031, 2032, 2033, 2034, 2035 |
| Número de execuções | 5 |
| Âncoras públicas de calibração | 128 |
| Dados públicos de ajuste por cliente | 128 |

Foram concluídas cinco execuções, cada uma com três rondas federadas. Em cada ronda, os 22 clientes participaram e os resultados foram recebidos sem falhas.

Cada execução contabilizou 66 liberações privadas (22 clientes × 3 rondas).

## 3. Contabilização da privacidade

O orçamento de ε = 0,8 corresponde a uma execução completa de três rondas, e não a cada ronda individualmente.

Considerando as cinco execuções e a composição sequencial dos orçamentos, o ε acumulado é aproximadamente 4,0 por cliente, assumindo que os mesmos dados de cada cliente são reutilizados e que os resultados das execuções são observáveis.

O registo de privacidade contabilizou 15 liberações privadas por cliente ao longo das cinco execuções. Não foram identificadas liberações não privadas nessas execuções.

A garantia documentada pelo mecanismo aplica-se ao canal de mensagens cliente-servidor de uma execução, sob as condições do protocolo e da configuração pública estabelecida. Não deve ser interpretada como uma garantia de privacidade para todos os artefactos ou resultados da investigação.

## 4. Resultados — DEV

Valores apresentados como média ± desvio-padrão amostral entre os cinco seeds.

| Dataset | Abordagem | PR-AUC | ROC-AUC | Recall | FAR | F1 |
|---|---|---:|---:|---:|---:|---:|
| icentia11k | global_student | 0,3345 ± 0,1326 | 0,5700 ± 0,2652 | 0,7667 ± 0,4346 | 0,7903 ± 0,4423 | 0,2773 ± 0,1552 |
| icentia11k | local_control | 0,5427 ± 0,0183 | 0,6936 ± 0,0098 | 0,5186 ± 0,0445 | 0,3097 ± 0,0631 | 0,3099 ± 0,0138 |
| icentia11k | local_teacher | 0,5471 ± 0,0132 | 0,7053 ± 0,0078 | 0,5336 ± 0,0074 | 0,3117 ± 0,0339 | 0,3102 ± 0,0092 |
| physionet2017 | global_student | 0,3780 ± 0,0380 | 0,4448 ± 0,0552 | 0,9255 ± 0,0924 | 0,9195 ± 0,0863 | 0,5576 ± 0,0196 |
| physionet2017 | local_control | 0,5740 ± 0,0314 | 0,6579 ± 0,0232 | 0,6157 ± 0,0852 | 0,4255 ± 0,0628 | 0,5300 ± 0,0461 |
| physionet2017 | local_teacher | 0,5980 ± 0,0404 | 0,6603 ± 0,0374 | 0,5261 ± 0,0844 | 0,3177 ± 0,0508 | 0,4966 ± 0,0570 |

## 5. Resultados — TRAIN

Valores apresentados como média ± desvio-padrão amostral entre os cinco seeds.

| Dataset | Abordagem | PR-AUC | ROC-AUC | Recall | FAR | F1 |
|---|---|---:|---:|---:|---:|---:|
| icentia11k | global_student | 0,3497 ± 0,0076 | 0,4255 ± 0,0643 | 0,7180 ± 0,4389 | 0,7160 ± 0,4396 | 0,3111 ± 0,1787 |
| icentia11k | local_control | 0,8258 ± 0,0195 | 0,8910 ± 0,0181 | 0,7402 ± 0,0405 | 0,0727 ± 0,0182 | 0,7217 ± 0,0352 |
| icentia11k | local_teacher | 0,8292 ± 0,0243 | 0,8935 ± 0,0176 | 0,7543 ± 0,0316 | 0,0599 ± 0,0180 | 0,7493 ± 0,0400 |
| physionet2017 | global_student | 0,4585 ± 0,0519 | 0,4858 ± 0,0747 | 0,9308 ± 0,0688 | 0,9158 ± 0,1026 | 0,5684 ± 0,0145 |
| physionet2017 | local_control | 0,6423 ± 0,0143 | 0,6758 ± 0,0235 | 0,6179 ± 0,0356 | 0,4140 ± 0,0874 | 0,5425 ± 0,0266 |
| physionet2017 | local_teacher | 0,6277 ± 0,0109 | 0,6522 ± 0,0153 | 0,5000 ± 0,1092 | 0,3211 ± 0,0885 | 0,4863 ± 0,0605 |

## 6. Observações preliminares

- Os resultados representam médias e desvios-padrão de cinco seeds; não constituem, por si só, comparações estatísticas pareadas.
- Na avaliação DEV, o global_student apresenta desempenho inferior aos modelos local_control e local_teacher nos dois datasets.
- No icentia11k, o global_student apresenta elevada variabilidade entre seeds.
- No physionet2017, o global_student apresenta recall elevado, mas também uma taxa de falsos alarmes (FAR) muito elevada.
- Estes resultados não permitem atribuir a degradação observada exclusivamente ao mecanismo LDP, uma vez que os baselines sem privacidade ainda não foram executados.

## 7. Limitações e trabalho pendente

1. Executar o baseline raw, sem clipping e sem ruído.
2. Executar o baseline clipped, com clipping e sem ruído.
3. Comparar os resultados sob o mesmo protocolo experimental e os mesmos seeds.
4. Avaliar a utilidade do modelo global e o impacto do ruído com base nos resultados completos.
5. Rever as conclusões após a comparação dos baselines.

As execuções completas dos baselines não estão incluídas neste relatório. Portanto, as observações devem ser tratadas como preliminares, sem conclusões definitivas sobre o impacto isolado da privacidade diferencial.

## 8. Testes da implementação

O ficheiro `support/model_tests/verify_ldp.py` foi executado com sucesso:

- 6 testes executados.
- 6 testes aprovados.
- Tempo de execução: 22,760 segundos.

Os testes cobrem clipping, limites de privacidade, invariância ao centramento, esgotamento do orçamento, composição do ledger e integração do fluxo de treino com logits ruidosos.

A aprovação dos testes valida os comportamentos cobertos pela suite, mas não substitui a avaliação experimental nem a comparação com os baselines.
