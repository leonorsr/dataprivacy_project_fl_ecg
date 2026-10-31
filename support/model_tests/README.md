# Apoio aos notebooks e verificação

shared_smoke.py é necessário aos três notebooks centralizados em Tests. Os resultados centralizados ficam em artifacts/test_models/. Esses testes podem manter RF/KNN como referências centrais; isso não os inclui no FL ativo.

verify_global.py verifica agregação FedAvg, exclusão dos candidatos incompatíveis, treino de um aluno CNN1D global por logits de professores heterogéneos, redistribuição exata dos pesos e deteção de broadcasts corrompidos. Usa apenas dados sintéticos temporários.

O verificador da exp5 anterior foi arquivado com os scripts de consenso sem aluno global. O protocolo ativo está em support/federated_global/ e run_global_experiments.py.
