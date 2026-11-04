"""Estudo offline dos logits guardados, sem treinar modelos nem aceder a dev/test.

Executar da raiz do repositório:
    python 'Privacy Techniques/LDP/experiencias_logits.py'
Os resultados vão exclusivamente para 'Privacy Techniques/LDP/resultados/'.
A aleatoriedade reproduzível é de SIMULAÇÃO; não é adequada para produção.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ldp_logits import (
    parametros_ruido, para_logits, recortar_por_cliente, representar,
    softmax_binario, validar_logits,
)

RAIZ = Path(__file__).resolve().parents[2]
FONTE_PADRAO = (RAIZ / "SoftLabels tests/final candidates/artifacts/confirmation"
                / "A_AE_CNN__MobileNetV2/2031")
SAIDA_PADRAO = Path(__file__).resolve().parent / "resultados"
EPSILONS_TOTAL = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
LIMITES = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0]
MECANISMOS = ["laplace", "gaussiano"]
REPRESENTACOES = ["original", "centrado", "margem"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def carregar_rondas(pasta):
    """Confere a origem, os pesos e o consenso guardado; não altera os ficheiros."""
    resultado = []
    for ronda in (1, 2, 3):
        caminho = pasta / f"teacher_round_{ronda}.npz"
        if not caminho.is_file():
            raise FileNotFoundError(f"Artefacto em falta: {caminho}")
        with np.load(caminho, allow_pickle=False) as dados:
            exigidas = {"logits", "probabilities", "consensus", "weights", "client_ids", "anchor_ids"}
            if not exigidas.issubset(dados.files):
                raise ValueError(f"Faltam chaves em {caminho}: {exigidas - set(dados.files)}")
            logits = validar_logits(dados["logits"])
            probabilidades = np.asarray(dados["probabilities"], dtype=np.float64)
            consenso = np.asarray(dados["consensus"], dtype=np.float64)
            pesos = np.asarray(dados["weights"], dtype=np.float64)
            clientes = np.asarray(dados["client_ids"]).copy()
            ancoras = np.asarray(dados["anchor_ids"]).copy()
        if logits.shape != (22, 32, 2) or probabilidades.shape != logits.shape:
            raise ValueError(f"Forma incompatível em {caminho}")
        if consenso.shape != (32, 2) or pesos.shape != (22,):
            raise ValueError(f"Consenso ou pesos incompatíveis em {caminho}")
        if len(clientes) != 22 or len(ancoras) != 32 or len(set(clientes)) != 22 or len(set(ancoras)) != 32:
            raise ValueError("Identificadores de clientes/âncoras inválidos")
        if not np.isfinite(pesos).all() or (pesos < 0).any() or not np.isclose(pesos.sum(), 1.0):
            raise ValueError("Os pesos guardados têm de ser finitos, não negativos e somar 1")
        if not np.isfinite(consenso).all() or not np.isfinite(probabilidades).all():
            raise ValueError("Probabilidades/consenso não finitos")
        nova_prob = softmax_binario(logits)
        novo_consenso = np.average(nova_prob, axis=0, weights=pesos)
        if not np.allclose(nova_prob, probabilidades, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Softmax não reproduz as probabilidades guardadas em {caminho}")
        if not np.allclose(novo_consenso, consenso, rtol=1e-6, atol=1e-7):
            raise ValueError(f"Média ponderada não reproduz o consenso guardado em {caminho}")
        if resultado:
            if not np.array_equal(clientes, resultado[0]["clientes"]):
                raise ValueError("Ordem dos clientes difere entre rondas")
            if not np.array_equal(ancoras, resultado[0]["ancoras"]):
                raise ValueError("Ordem das âncoras difere entre rondas")
            if not np.allclose(pesos, resultado[0]["pesos"]):
                raise ValueError("Pesos diferem entre rondas")
        resultado.append(dict(ronda=ronda, logits=logits, prob=probabilidades,
                              consenso=consenso, pesos=pesos, clientes=clientes,
                              ancoras=ancoras, fonte=str(caminho.relative_to(RAIZ))
                              if caminho.is_relative_to(RAIZ) else str(caminho),
                              sha256=sha256(caminho)))
    return resultado


def estatisticas(prob, original, consenso_original, pesos):
    """Aceita forma (repetições, clientes, âncoras, classes)."""
    if prob.ndim != 4:
        raise ValueError("Esperada dimensão de repetições")
    consenso = np.einsum("rcak,c->rak", prob, pesos, optimize=True)
    diff_prob = prob[..., 1] - original[None, ..., 1]
    diff_consenso = consenso[..., 1] - consenso_original[None, ..., 1]
    decisao_original = np.argmax(consenso_original, axis=-1)
    decisao_nova = np.argmax(consenso, axis=-1)
    normal_anomalia = np.sum((decisao_original[None, :] == 0) & (decisao_nova == 1), axis=1)
    anomalia_normal = np.sum((decisao_original[None, :] == 1) & (decisao_nova == 0), axis=1)
    decisoes_locais_originais = np.argmax(original, axis=-1)
    decisoes_locais_novas = np.argmax(prob, axis=-1)
    locais_normal_anomalia = np.sum((decisoes_locais_originais[None, :, :] == 0) & (decisoes_locais_novas == 1), axis=(1, 2))
    locais_anomalia_normal = np.sum((decisoes_locais_originais[None, :, :] == 1) & (decisoes_locais_novas == 0), axis=(1, 2))
    return {
        "prob_mae": np.mean(np.abs(diff_prob), axis=(1, 2)),
        "consensus_mae": np.mean(np.abs(diff_consenso), axis=1),
        "consensus_mse": np.mean(diff_consenso ** 2, axis=1),
        "decision_flips": normal_anomalia + anomalia_normal,
        "normal_anomalia": normal_anomalia,
        "anomalia_normal": anomalia_normal,
        "local_flips": np.sum(decisoes_locais_novas != decisoes_locais_originais[None, :, :], axis=(1, 2)),
        "local_normal_anomalia": locais_normal_anomalia,
        "local_anomalia_normal": locais_anomalia_normal,
    }


def resumo_metricas(valores):
    saida = {}
    for nome, arr in valores.items():
        a = np.asarray(arr, dtype=np.float64)
        saida[f"{nome}_mean"] = float(np.mean(a))
        saida[f"{nome}_sd"] = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
        saida[f"{nome}_se_mean"] = float(np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
        saida[f"{nome}_p025"] = float(np.quantile(a, 0.025))
        saida[f"{nome}_p975"] = float(np.quantile(a, 0.975))
    return saida


def executar(pasta_fonte, pasta_saida, repeticoes, semente, delta_total):
    if repeticoes < 1:
        raise ValueError("repeticoes deve ser positivo")
    if not 0 < delta_total < 1:
        raise ValueError("delta total deve estar entre 0 e 1")
    rondas = carregar_rondas(pasta_fonte)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semente)  # repetibilidade do estudo; NÃO usar em produção
    resumos, por_ronda, baseline = [], [], []
    n_combinacoes = len(MECANISMOS) * len(REPRESENTACOES) * len(EPSILONS_TOTAL) * len(LIMITES)
    contador = 0

    for mecanismo in MECANISMOS:
        ordem = 1 if mecanismo == "laplace" else 2
        delta_usado = 0.0 if mecanismo == "laplace" else delta_total
        for representacao in REPRESENTACOES:
            # Pré-calcular o recorte sem ruído, por ronda e por C.
            preparados = {}
            for limite in LIMITES:
                por_limite = []
                for ronda in rondas:
                    valores = representar(ronda["logits"], representacao)
                    cortado, flags = recortar_por_cliente(valores, limite, ordem)
                    prob_corte = softmax_binario(para_logits(cortado, representacao))[None, ...]
                    so_corte = estatisticas(prob_corte, ronda["prob"], ronda["consenso"], ronda["pesos"])
                    por_limite.append((cortado, float(flags.mean()), so_corte))
                    baseline.append(dict(mechanism=mecanismo, representation=representacao, clip_C=limite,
                                         round=ronda["ronda"], clip_fraction=float(flags.mean()),
                                         **{k: float(v[0]) for k, v in so_corte.items()}))
                preparados[limite] = por_limite

            for epsilon_total in EPSILONS_TOTAL:
                epsilon_ronda = epsilon_total / len(rondas)
                delta_ronda = delta_usado / len(rondas)
                for limite in LIMITES:
                    _, sensibilidade, escala = parametros_ruido(mecanismo, limite, epsilon_ronda, delta_ronda)
                    metrica_por_ronda, fracoes = [], []
                    for ronda, (cortado, fracao, _) in zip(rondas, preparados[limite]):
                        tamanho = (repeticoes,) + cortado.shape
                        if mecanismo == "laplace":
                            ruido = rng.laplace(0.0, escala, size=tamanho)
                        else:
                            ruido = rng.normal(0.0, escala, size=tamanho)
                        privados = para_logits(cortado[None, ...] + ruido, representacao)
                        probabilidades = softmax_binario(privados)
                        metricas = estatisticas(probabilidades, ronda["prob"], ronda["consenso"], ronda["pesos"])
                        metrica_por_ronda.append(metricas)
                        fracoes.append(fracao)
                        por_ronda.append(dict(mechanism=mecanismo, representation=representacao,
                            epsilon_total=epsilon_total, epsilon_round=epsilon_ronda,
                            delta_total=delta_usado, delta_round=delta_ronda,
                            clip_C=limite, sensitivity=sensibilidade, noise_scale=escala,
                            round=ronda["ronda"], replicates=repeticoes, clip_fraction_mean=fracao,
                            **resumo_metricas(metricas)))
                    # Cada repetição é a média das 3 rondas; quantis descrevem
                    # dispersão do ruído na média de rondas, não intervalo clínico.
                    medias = {nome: np.mean(np.stack([m[nome] for m in metrica_por_ronda]), axis=0)
                              for nome in metrica_por_ronda[0]}
                    resumos.append(dict(mechanism=mecanismo, representation=representacao,
                        epsilon_total=epsilon_total, epsilon_round=epsilon_ronda,
                        delta_total=delta_usado, delta_round=delta_ronda, clip_C=limite,
                        sensitivity=sensibilidade, noise_scale=escala, rounds=len(rondas),
                        replicates=repeticoes, clip_fraction_mean=float(np.mean(fracoes)),
                        **resumo_metricas(medias)))
                    contador += 1
                    if contador % 42 == 0 or contador == n_combinacoes:
                        print(f"Progresso: {contador}/{n_combinacoes} configurações", flush=True)

    tabela = pd.DataFrame(resumos)
    tabela.to_csv(pasta_saida / "resumo.csv", index=False)
    pd.DataFrame(por_ronda).to_csv(pasta_saida / "por_ronda.csv", index=False)
    pd.DataFrame(baseline).to_csv(pasta_saida / "somente_clipping.csv", index=False)
    # Seleção exploratória: escolher C por método e orçamento e depois ordenar
    # por MAE do consenso. Não é validação independente nem prova de DP.
    melhores_por_metodo = (tabela.sort_values(["consensus_mae_mean", "clip_C"])
        .groupby(["epsilon_total", "mechanism", "representation"], as_index=False).first())
    melhores_por_metodo = melhores_por_metodo.sort_values(["epsilon_total", "consensus_mae_mean", "mechanism", "representation"])
    melhores_por_metodo.to_csv(pasta_saida / "melhor_C_por_metodo.csv", index=False)
    top3 = melhores_por_metodo.groupby("epsilon_total", sort=True).head(3)
    top3.to_csv(pasta_saida / "top3_por_epsilon.csv", index=False)
    manifesto = {
        "fonte": [dict(round=r["ronda"], path=r["fonte"], sha256=r["sha256"]) for r in rondas],
        "branch_base": "testing-models", "checkpoint": "A_AE_CNN__MobileNetV2/2031",
        "inputs": "22 clientes x 32 âncoras sintéticas x 2 classes por ronda",
        "classes": {"0": "normal", "1": "anomalia"},
        "consenso": "softmax com temperatura 1 por cliente/âncora; média com pesos guardados",
        "rondas": len(rondas), "epsilons_totais": EPSILONS_TOTAL,
        "limites_C": LIMITES, "mecanismos": MECANISMOS,
        "representacoes": REPRESENTACOES, "delta_total_gaussiano": delta_total,
        "delta_total_laplace": 0, "repeticoes_por_configuracao_e_ronda": repeticoes,
        "semente_publica_de_simulacao": semente,
        "selecao": "exploratória nos mesmos logits, minimiza MAE médio do consenso por epsilon",
        "avisos": ["Não é uma execução federada em ciclo fechado; logits de rondas posteriores são históricos.",
                   "Logits originais já estão guardados no repositório; LDP não desfaz divulgações passadas.",
                   "Não há rótulos verdadeiros das âncoras; não inferir PR-AUC, recall ou FAR clínicos.",
                   "Semente conhecida e NumPy RNG apenas para simulação, sem garantia de produção.",
                   "Métricas e seleção consultam logits não protegidos e podem ser sensíveis."]
    }
    (pasta_saida / "configuracao.json").write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n")
    print(f"Resultados guardados em: {pasta_saida}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fonte", type=Path, default=FONTE_PADRAO)
    parser.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    parser.add_argument("--repeticoes", type=int, default=200)
    parser.add_argument("--semente", type=int, default=2031)
    parser.add_argument("--delta-total", type=float, default=1e-5)
    args = parser.parse_args()
    executar(args.fonte.resolve(), args.saida.resolve(), args.repeticoes, args.semente, args.delta_total)


if __name__ == "__main__":
    main()
