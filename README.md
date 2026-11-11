# Deteção de arritmias: aprendizagem federada e privacidade

Projeto de comparação de modelos de deteção de arritmias nos datasets **Icentia11k** e **PhysioNet 2017**, com testes centralizados, FedAvg, distilação federada por soft labels e mecanismos de privacidade. Este README concentra a documentação das pastas do projeto.

## Organização

| Pasta | Conteúdo e finalidade |
|---|---|
| `datasets/` | Dados originais, amostras e versões preparadas dos dois datasets. Inclui partições de treino, dev e test e dados organizados por cliente. Os dados locais estão excluídos do Git. |
| `DataPreparation/` | Notebooks de preparação de cada dataset: limpeza, transformação dos sinais e organização dos dados usados nas experiências. |
| `DataPreparation/FeatureExtraction/` | Extração de características, construção dos datasets com features e respetivos artefactos locais. |
| `EDA/Icentia11k/` e `EDA/Physionet2017/` | Análise exploratória de cada origem, distribuição dos dados e características dos sinais. |
| `Tests/` | Testes centralizados nos notebooks `test_models_icentia11k.ipynb` e `test_models_physionet.ipynb`; comparação em `comparation_test_models.ipynb`. RF/KNN podem ser referências centralizadas, sem serem candidatos do FL ativo. |
| `FedAvg tests/` | Experiências de FedAvg: treino local, agregação dos parâmetros ponderada pelos exemplos de fit e redistribuição do modelo global. O notebook `comparation_fedavg.ipynb` reúne métricas e rankings. |
| `FedAvg tests/common/` | Módulos partilhados de dados, modelos, aprendizagem e métricas. |
| `SoftLabels tests/` | Experiências de distilação federada, comparando combinações de modelos locais com um modelo global CNN1D. O notebook `comparation_softlabels.ipynb` reúne as experiências e comparações. |
| `SoftLabels tests/common/` | Componentes de apoio às experiências de soft labels. |
| `SoftLabels tests/final candidates/` | Confirmação das candidatas A e B: A usa AE+CNN no Icentia e MobileNetV2 no PhysioNet; B usa CNN1D no Icentia e MobileNetV2 no PhysioNet. Ambas usam CNN1D global. A configuração de confirmação usa cinco seeds, três rondas e dez épocas locais por ronda. Resultados em `resultados.ipynb`. |
| `SoftLabels tests/final candidates/ldp_inputs/fit_weighted_v2/` | Logits guardados e metadados de proveniência da configuração ponderada, usados nos estudos de mensagens. Cada ficheiro de ronda contém 22 clientes, 32 âncoras e duas classes. |
| `SoftLabels tests/moments_comparison/` | Artefactos da comparação entre probabilidades de anomalia e representações baseadas em média/desvio-padrão, incluindo MC Dropout. |
| `SoftLabels tests/experiment_overview/` | Resultados de apoio à comparação entre experiências. |
| `Privacy Techniques/LDP/` | Estudo exploratório **isolado dos logits guardados**, sem treinar modelos nem executar FL. Um único notebook integra experiências, comparação e smoke tests do compromisso privacidade–utilidade. |
| `Privacy Techniques/LDP + soft labels/` | LDP integrada no ciclo de treino federado: clipping e ruído nas mensagens dos clientes antes da agregação. Notebooks de treino e comparação com referências sem LDP e controlos apenas com clipping. |
| `Privacy Techniques/LDP + SA/` | Experiência conjunta P2 (Gaussiano/margem/C=0,5, ε=4, δ=10⁻⁵) + SA (soma única, 24 bits, t=12). Cinco seeds, três rondas, dez épocas locais; avaliação final do modelo global em todos os exemplos elegíveis de test após congelar os treinos. Notebook `experiencias_ldp_sa.ipynb`. |
| `Privacy Techniques/SA/` | Área de Secure Aggregation. A implementação e os notebooks disponíveis dependem da branch; o estudo SA compara o acesso a mensagens individuais com o acesso apenas ao agregado. |
| `Privacy Techniques/SA/Communication/` | Área de comparação de comunicação e modelos com/sem SA, com subpastas FedAvg e Softlabels nas branches que contêm esse estudo. |
| `support/` | Código de apoio, geração de relatórios, gráficos e validações; mantém os detalhes de implementação fora das pastas de notebooks. |
| `support/model_tests/` | Apoio aos testes centralizados e verificadores de agregação, redistribuição, decisões e mecanismos de privacidade. |
| `support/federated_global/` | Implementação dos protocolos FedAvg e soft labels, avaliação, validação e geração dos notebooks. |
| `support/ldp_logits/` | Implementação dos estudos atuais de LDP nos logits: simulação isolada, smoke tests, mecanismos do treino, contabilização do orçamento e atualização dos notebooks. |
| `support/ldp_sa/` | Treino conjunto LDP + SA, protocolo SA extraído com proveniência, testes funcionais, avaliador de test em blocos e geração do notebook. |
| `support/ldp/` | Implementação de apoio de uma etapa anterior do estudo LDP; para o estudo atual dos logits usar `support/ldp_logits/`. |
| `artifacts/` e subpastas `artifacts/` | Outputs locais: caches, checkpoints, previsões, métricas, figuras, logs e históricos. Não representam novas experiências por si só. Em geral estão excluídos do Git. |
| `.vscode/` | Configuração local do editor. |
| `.venv/` | Ambiente virtual local, excluído do Git. |

## Protocolos e avaliação

**FedAvg:** o servidor distribui um modelo compatível, os clientes treinam localmente e o servidor agrega parâmetros com pesos proporcionais ao fit. O modelo atualizado é redistribuído em cada ronda. RF/KNN e GRU não participam nos candidatos ativos deste protocolo.

**Soft labels:** os modelos locais produzem logits sobre 32 âncoras sintéticas de referência alinhadas entre origens. O servidor converte-os em probabilidades, calcula o consenso ponderado e treina a CNN1D global. O modelo global completo é redistribuído e orienta os modelos locais nas rondas seguintes. As âncoras são referências artificiais, sem rótulos clínicos, e não substituem os exemplos reais de avaliação.

Na configuração ponderada atual de 22 clientes, cada cliente tem 128 exemplos de fit: **16 clientes Icentia contribuem com 72,73% e seis clientes PhysioNet com 27,27%**. Usam-se os pesos guardados nos ficheiros; estes valores não são uma regra fixa para outras partições. Resultados históricos podem corresponder a protocolos diferentes e devem ser interpretados segundo os seus metadados.

O critério atual privilegia o **PR-AUC dos modelos locais**, com avaliação no holdout do próprio cliente e média ponderada pelos exemplos de fit. O modelo global é avaliado separadamente. Dev é uma avaliação complementar; test permanece reservado. As seeds medem variabilidade do treino com os dados fixados, não novas amostras de pacientes. Rankings e decisões exploratórias não demonstram desempenho clínico independente.

## Estudos de privacidade

### LDP isolada dos logits

Em `Privacy Techniques/LDP/`:

O único notebook, `experiencias_logits.ipynb`, está organizado em três secções:

1. Dados e estudo completo: grelha exploratória de mecanismos, representações, clipping e ε.
2. Verificação curta: três perfis fixos, ε total em {0,5; 1; 2; 4; 8; 16} e 20 sorteios por perfil, ε e mensagem histórica; gráficos de MAE e decisões preservadas.
3. Comparação e interpretação: seleção exploratória, efeito do clipping e limites da análise.

O estudo isolado usa os três ficheiros históricos da configuração A, seed 2031. Não inicia treinos nem novas rondas. Menor ε corresponde a maior proteção mantendo o mecanismo e δ. Laplace usa δ=0; o Gaussiano usa δ total=10⁻⁵. A utilidade aqui mede fidelidade às previsões originais: **não é PR-AUC, recall ou FAR clínico**, pois as âncoras não têm rótulos clínicos. O RNG reproduzível destina-se à simulação.

### LDP com soft labels

Em `Privacy Techniques/LDP + soft labels/`, a experiência usa ε total=4 por execução de um cliente, cinco seeds, três rondas e dez épocas locais por ronda. Compara referência sem LDP, controlos só com clipping e três perfis com ruído: Gaussiano/logits originais/C=0,5; Gaussiano/margem/C=0,5; Laplace/margem/C=2. São 35 avaliações planeadas, incluindo cinco referências compatíveis reutilizadas após validação.

O clipping abrange a mensagem completa do cliente. O ruído é aplicado antes da agregação e afeta os treinos seguintes. Os mecanismos do treino usam `diffprivlib` com aleatoriedade segura. O orçamento é dividido pelas três rondas e o ledger compõe divulgações de diferentes tentativas, seeds e perfis: **ε=4 não é o orçamento da grelha inteira**. Os controlos sem ruído não são privados.

A garantia estudada refere-se às mensagens preparadas sob transformações fixadas. A normalização histórica usa estatísticas agregadas do fit; não foi demonstrada privacidade de ponta a ponta para todo o processamento do ECG bruto. Métricas, seleção de parâmetros e checkpoints não ficam automaticamente protegidos. LDP não protege retroativamente logits já divulgados.

### Secure Aggregation

SA permite ao servidor obter o agregado sem observar cada contribuição individual, dentro das hipóteses do protocolo. Não substitui LDP nem impede, por si só, envenenamento por clientes maliciosos ou inferência a partir do agregado. Os estudos presentes noutras branches avaliam também comunicação, custos e utilidade dos modelos; consultar os notebooks dessas branches para os resultados específicos.

## Execução

Executar a partir da raiz, com as dependências de `requirements.txt` instaladas no ambiente do projeto.

```bash
# Piloto e experiências dos protocolos globais
.venv/bin/python run_global_experiments.py --pilot
.venv/bin/python run_global_experiments.py

# Validação e relatórios desses protocolos
.venv/bin/python run_global_experiments.py --phase validate
.venv/bin/python run_global_experiments.py --phase report

# Smoke test LDP isolado; atualiza o respetivo notebook
.venv/bin/python -m support.ldp_logits.smoke

# Repetir o estudo isolado completo; não inicia FL
.venv/bin/python -m support.ldp_logits.offline --repeticoes 200

# Atualizar apenas os notebooks do estudo isolado
.venv/bin/python -m support.ldp_logits.offline_notebooks

# Treino conjunto LDP + SA, seguido de avaliação global em test
.venv/bin/python -m support.ldp_sa.experiment

# Treino federado com LDP e atualização dos seus notebooks
.venv/bin/python -m support.ldp_logits.run
```

Os outputs do estudo isolado ficam em `artifacts/ldp_logits/offline/`. O estado do treino integrado está em `artifacts/ldp_logits/closed_loop/status.json`, os estados por ronda em `artifacts/ldp_logits/closed_loop/runs/{condicao}/status.json` e o log em `artifacts/ldp_logits/run.log`. Os notebooks incorporam tabelas e figuras para consulta. As configurações e assinaturas guardadas nos artefactos são a referência para identificar cada execução.

O estudo conjunto LDP + SA guarda estado e logs em `artifacts/ldp_sa/`. A configuração é congelada em `frozen_design.json`; o test só é aberto após validar as cinco seeds. Os clientes de test são novos: não se apresenta a transferência de um modelo local de outro cliente como avaliação local. O notebook explicita que a implementação SA é semi-honesta e não fornece defesa contra envenenamento nem segurança completa perante participantes ativamente maliciosos.
