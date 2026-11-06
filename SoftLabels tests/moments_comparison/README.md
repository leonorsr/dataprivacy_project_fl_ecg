# Média e desvio-padrão por âncora com MC Dropout

Comparação em `../comparation_softlabels.ipynb` e `experiencias_moments.ipynb`.

Quatro variantes: 64 probabilidades atuais; 32 probabilidades de anomalia (normal = 1 − anomalia); 32 médias de anomalia sobre 20 passagens MC Dropout; 32 médias + 32 desvios-padrão dessas passagens, por âncora. São quatro condições em cinco seeds (20 comparações), com três rondas e dez épocas locais, candidata A e CNN1D global. Pesos de fit 72,73%/27,27%. Os pares MC usam as mesmas sementes e só Dropout fica ativo; BatchNorm permanece eval e não atualiza estatísticas.

O alvo continua associado à mesma âncora e à classe anomalia. A média não garante reproduzir cada decisão determinística e não preserva toda a distribuição. A variância agregada soma os segundos momentos e subtrai o quadrado da média, incluindo variabilidade dentro do professor e desacordo entre professores. No aluno, a variante média+std pondera a entropia cruzada por `1/(1+4*var)`, normalizada para média 1. A variante só média MC isola o efeito do uso do std. O coeficiente 4 é fixado a partir da variância máxima 1/4 de uma probabilidade, sem ajuste ao holdout. A incerteza não é tratada como confiança calibrada.

O critério principal é AP local no holdout do próprio cliente, ponderada pelo fit, com diferenças pareadas nas cinco seeds e última ronda fixa. AP global, comunicação e cálculo local são secundários. Test não é usado. Sem SA/LDP nesta ablação; SA futura exige momentos aditivos preparados nos clientes. O payload média+std contém os mesmos 64 números da referência: não há redução face a esta. O download do modelo global também é contado.

As cinco referências determinísticas compatíveis em SA/Communication são reutilizadas apenas após nova validação; sem referências, a condição atual também é treinada. `reference_reuse.json` regista as origens. As outras 15 condições exigem novos treinos. Resultados e gráficos são atualizados quando uma condição completa é validada. Artefactos em `{representação}/artifacts/` são ignorados pelo Git.

A experiência anterior de dois números por cliente foi interrompida após a clarificação do utilizador e preservada em `artifacts/softlabels_moments_history/client_wide_v1/`, sem misturar resultados. Correr na raiz: `.venv/bin/python "SoftLabels tests/moments_comparison/run.py"`.
