# Mensagens atualizadas para os testes de SA

Usar `fit_weighted_v2/confirmation/{candidata}/{seed}/teacher_round_{1,2,3}.npz`.

São mensagens reais do novo treino, não uma alteração manual dos pesos antigos. Incluem logits, probabilidades, consenso, pesos, contagens de fit, IDs de clientes e âncoras. Existem A e B, seeds 2031–2035, três rondas (30 ficheiros). O carregamento de `protocol.py` aponta para esta pasta, por omissão A/2031.

Cada cliente tem 128 exemplos de fit e peso 1/22. Icentia: 16 clientes, 2048 exemplos, 72,73%. PhysioNet: 6 clientes, 768 exemplos, 27,27%. A candidata selecionada pelo novo treino foi A.

`manifest.json` regista commit de origem e SHA-256 de cada ficheiro. Os run_config e audits acompanham cada execução; design, contagens e validação estão na raiz do pacote.

O notebook SA foi adaptado para usar estes pesos em ambos os modos e na referência sem SA. Os pesos são renormalizados entre os clientes que enviaram mensagens mascaradas, incluindo quem não participou no desmascaramento se houver respostas suficientes para reconstruir o agregado. Os testes antigos de SA foram preservados em artifacts/sa_support/history/before_fit_weighted_sa/. As execuções atuais são registadas em artifacts/sa_support/execution_status.json.
