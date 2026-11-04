# LDP nos logits já guardados

Esta pasta contém um estudo **offline e exploratório** da privacidade diferencial local aplicada a logits. Não treina modelos, não usa conjuntos dev/test para ajuste, não altera a arquitetura federada e não implementa agregação segura. Os arquivos originais de outras pastas não foram modificados.

## 1. Quais são os dados

O estudo usa apenas os três arquivos já existentes:

```text
SoftLabels tests/final candidates/artifacts/confirmation/
  A_AE_CNN__MobileNetV2/2031/teacher_round_1.npz
  A_AE_CNN__MobileNetV2/2031/teacher_round_2.npz
  A_AE_CNN__MobileNetV2/2031/teacher_round_3.npz
```

A grafia efetiva tem **dois sublinhados** antes de `MobileNetV2`. Cada arquivo contém `logits` com forma `(22, 32, 2)`: 22 clientes, 32 âncoras sintéticas e duas classes. Os pesos guardados atribuem metade do peso à origem Icentia11k e metade à PhysioNet 2017. Dentro de cada origem os clientes têm o mesmo peso: 1/32 para cada um dos 16 clientes Icentia e 1/12 para cada um dos seis PhysioNet.

O arquivo contém também `probabilities`, `consensus`, `weights`, `client_ids` e `anchor_ids`. Antes de experimentar, o programa confirma formas, identidades, pesos e se a reconstrução abaixo coincide com o consenso salvo:

1. Aplicar softmax a cada par de logits, com temperatura 1.
2. Calcular a média das probabilidades usando os **pesos guardados**, por âncora e classe.

É a mesma regra usada em `support/federated_global/soft.py`, linhas 85–107. Não é média simples de 22 clientes nem softmax da média dos logits. O estudante CNN1D é um modelo treinado na arquitetura original, mas este estudo **não o treina nem altera**.

Os arquivos têm logits de uma execução anterior sem LDP. As âncoras são sintéticas e não têm rótulos clínicos no NPZ. Os resultados indicam fidelidade ao consenso original, não capacidade de diagnóstico.

## 2. Como executar

Executa a partir da **raiz do repositório**, onde existe `requirements.txt`. São necessárias as bibliotecas Python `numpy`, `pandas`, `scipy` e `matplotlib`, já presentes na lista geral de dependências do projeto.

```bash
# 1. Testes matemáticos, de formato e de reprodução do consenso.
python -m unittest discover -s 'Privacy Techniques/LDP' -p 'testar_ldp_logits.py' -v

# 2. Experiências sobre os três arquivos originais; 200 sorteios por condição.
python 'Privacy Techniques/LDP/experiencias_logits.py' --repeticoes 200

# 3. Gráficos a partir dos CSV gerados.
python 'Privacy Techniques/LDP/graficos.py'
```

No Windows, em PowerShell, os comandos também podem usar as aspas simples indicadas. Usa `python3` se for esse o nome do Python no teu sistema. A simulação usa uma semente reproduzível **apenas para investigação**; não copies essa política de aleatoriedade para uma implantação real.

O programa aceita `--fonte` para indicar outra pasta com os três `teacher_round_*.npz`, `--saida` para indicar outra pasta de resultados, `--repeticoes`, `--semente` e `--delta-total`. A configuração e os hashes SHA-256 dos arquivos são guardados em `resultados/configuracao.json`.

## 3. O que está dentro desta pasta

- `ldp_logits.py`: operações matemáticas para representar logits, recortar por cliente e simular Laplace/Gaussiano. A calibragem gaussiana resolve a equação analítica em log-escala.
- `experiencias_logits.py`: carrega os NPZ existentes, confirma o consenso, executa a grelha de experiências e gera tabelas. **Não executa qualquer treino.**
- `testar_ldp_logits.py`: testes de clipping, sensibilidade, calibração, representações e reconstrução dos três consensos.
- `graficos.py`: gera figuras a partir das tabelas já criadas.
- `resultados/`: configuração, tabelas e imagens.
- `RELATORIO_6_2.md`: explica os resultados observados e as limitações.

Existe também código anterior em `support/ldp/`, fora desta pasta. Não foi alterado. O seu fluxo de experiências pode envolver treino em ciclo fechado e, por isso, não é o mesmo que este teste offline dos logits congelados.

## 4. Onde entra a LDP

Em cada ronda, um cliente recebe as 32 âncoras e calcula 32 pares de logits. Antes de enviar qualquer valor ao servidor, aplica-se uma transformação aos **64 valores do cliente em conjunto**; a variante de margem transforma-os primeiro em 32 diferenças binárias. Segue-se clipping e depois adição de ruído. Só a mensagem perturbada deve sair do cliente.

O presente código aplica essa operação retrospectivamente aos arrays já guardados; não está ligado ao cliente real. Quando a equipa integrar uma opção na arquitetura, a chamada terá de correr no cliente antes do envio. O servidor converte a mensagem privada em probabilidades e calcula o consenso ponderado. Na arquitetura real, o gerador de ruído, a contabilidade, os erros, metadados e todas as outras mensagens têm de ser revistos.

### Representações

- `original`: os 32 pares de logits tal como foram guardados.
- `centrado`: subtrai-se a média de cada par; isso não altera a softmax antes do clipping porque só elimina um deslocamento comum às duas classes.
- `margem`: guarda-se a diferença `logit_classe_1 - logit_classe_0` por âncora. Para converter o resultado privado de volta a dois logits, usa-se `[-margem/2, margem/2]`. Antes de clipping/ruído, a softmax é igual à original.

O recorte é feito **depois** da transformação escolhida, sobre o vetor completo de cada cliente. O mesmo número C em representações diferentes não implica igual intensidade de clipping.

### Sensibilidade, ruído e composição

A unidade proposta é todo o vetor de um cliente participante, sob substituição arbitrária e com participação e parâmetros públicos. Depois de clipping da norma global a `C`, duas saídas recortadas podem diferir em até `2C` na norma correspondente.

- `laplace`: clipping da norma **L1**; sensibilidade L1 `2C`; ruído independente por coordenada com escala `2C / epsilon_ronda`; `delta=0`.
- `gaussiano`: clipping da norma **L2**; sensibilidade L2 `2C`; ruído Gaussiano independente por coordenada com sigma calculado pela calibração Gaussiana analítica. O parâmetro `delta` é positivo e está indicado nas tabelas.

Não confundas L1 com L2. Para Gaussianos, a calibração usa a equação analítica do mecanismo, permitindo também epsilon por ronda maior do que 1; a fórmula clássica `sqrt(2 ln(1.25/delta))/epsilon` não é extrapolada para esses valores. A equação é resolvida por bisseção com `scipy.special.log_ndtr`; consulte Balle e Wang, *Improving the Gaussian Mechanism for Differential Privacy*, https://proceedings.mlr.press/v80/balle18a.html.

A grelha usa `epsilon_total` em `{0.5, 1, 2, 4, 8, 16}` para um ciclo de três mensagens por cliente. O orçamento por ronda é `epsilon_total / 3`. Para Gaussiano, `delta_total=10^-5` e `delta_ronda=delta_total/3`; para Laplace, `delta=0`. A soma é uma **cota de composição simples**, não uma afirmação de que todo o sistema já está protegido. Epsilon 8 ou 16 representa uma proteção muito mais fraca do que valores menores; é incluído para observar a troca privacidade–utilidade, não como recomendação.

Os limites `C` são `0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048`. Avaliam-se 2 mecanismos × 3 representações × 6 orçamentos × 14 valores de C = **504 configurações**, nas três rondas. A escolha da grelha e dos melhores C a partir dos mesmos logits é exploratória, não seleção independente nem procedimento automaticamente protegido por DP.

## 5. Como ler as tabelas

`resultados/resumo.csv` tem uma linha por configuração, com a média das três rondas e das repetições. `resultados/por_ronda.csv` mantém uma linha por ronda e configuração. `resultados/somente_clipping.csv` mostra o efeito determinístico de recortar sem ruído: **é um controlo não privado**, não um mecanismo LDP.

Para cada repetição:

- `prob_mae`: média de `|p_privada(classe 1) - p_original(classe 1)|` sobre os 22 × 32 = 704 pares cliente–âncora.
- `consensus_mae`: média do erro absoluto na probabilidade da classe 1 do consenso, nas 32 âncoras.
- `consensus_mse`: média dos erros quadráticos da mesma probabilidade, nas 32 âncoras.
- `decision_flips`: número, entre 32, de decisões do consenso que mudaram pelo critério `argmax` usado pelo projeto. Classe 0 é normal, classe 1 é anomalia; um empate vai para a classe 0.
- `normal_anomalia` e `anomalia_normal`: direções da alteração do consenso. São **discordâncias com a saída anterior**, não falsos positivos ou falsos negativos clínicos.
- `local_flips`: número, entre 704, de decisões locais diferentes. `local_normal_anomalia` e `local_anomalia_normal` separam os sentidos.
- `clip_fraction_mean`: fração média de clientes afetados pelo recorte. Em cenários privados, esta fração também é informação sensível.

`*_mean` é a média dos sorteios; `*_sd` o desvio dos sorteios; `*_se_mean` o erro-padrão Monte Carlo da média; `*_p025` e `*_p975` são os quantis 2,5% e 97,5% dos sorteios. Esses quantis descrevem apenas a variação do ruído para **os mesmos logits congelados**; não são um intervalo clínico nem intervalo de generalização para novos pacientes.

`melhor_C_por_metodo.csv` escolhe o C com menor `consensus_mae_mean` para cada mecanismo, representação e epsilon. `top3_por_epsilon.csv` ordena as três melhores combinações por esse mesmo critério para cada orçamento. A seleção é descritiva e pode mudar com outros logits, sementes, métricas ou repetições. Confere também MSE, mudanças de decisão, `epsilon`, `delta` e clipping antes de propor uma configuração para a arquitetura final.

## 6. O que NÃO se pode concluir

**PR-AUC, recall e FAR não podem ser calculados a partir destes logits isolados**, pois não há rótulos verdadeiros das âncoras. As métricas históricas de dev noutros arquivos são de uma execução sem esta LDP e não provam o efeito do ruído. Para estudar essas métricas será necessário um ensaio posterior apropriado da arquitetura completa, com a avaliação separada e autorizada.

As três rondas históricas foram geradas sem este ruído. Perturbá-las depois não recria o que aconteceria se, na ronda anterior, o estudante e os clientes já tivessem recebido um consenso privado. Não confundir este teste de canal com uma simulação federada em ciclo fechado.

Os logits originais já estão no repositório. Acrescentar ruído agora **não desfaz divulgações passadas**. Além disso, na preparação histórica algumas transformações dependem de estatísticas agregadas dos dados de treino; a troca de um cliente pode afetar outras mensagens. A garantia de todo o pipeline depende desse pré-processamento, de todas as mensagens e da composição entre experiências, não apenas da fórmula do ruído. Identidade/participação, métricas, logs, pesos dependentes dos dados, checkpoints e outras saídas podem constituir canais separados.

Por fim, este programa usa `numpy.random` com semente pública para pesquisa. **Não é uma implementação segura para dados clínicos reais**: a garantia matemática ideal não se transfere automaticamente para um gerador não criptográfico, precisão finita, erros observáveis ou divulgação de sementes. Use uma biblioteca auditada e uma análise formal para produção. Publicar a tabela de resultados ou escolher parâmetros com base em dados privados também pode divulgar informação e deve ser avaliado separadamente.
