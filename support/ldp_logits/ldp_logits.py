"""Experiências offline de LDP em logits binários já guardados.

ATENÇÃO: pesquisa/reprodução, não implementação de produção. Usa NumPy RNG
com semente pública e não contabiliza liberações reais. Não enviar a servidores
logits brutos, sementes, diagnósticos privados ou estatísticas não protegidas.
A unidade de clipping é o vetor COMPLETO das 32 âncoras de cada cliente.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import log_ndtr


def validar_logits(logits):
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 3 or z.shape[1:] != (32, 2) or z.shape[0] < 1:
        raise ValueError("Esperado array (clientes, 32 âncoras, 2 classes)")
    if not np.isfinite(z).all():
        raise ValueError("Os logits têm de ser finitos")
    return z


def representar(logits, representacao):
    """Transformação local: original, centrado por âncora ou margem binária."""
    z = validar_logits(logits)
    if representacao == "original":
        return z.copy()
    if representacao == "centrado":
        return z - z.mean(axis=-1, keepdims=True)
    if representacao == "margem":
        return z[:, :, 1] - z[:, :, 0]
    raise ValueError("Representação deve ser original, centrado ou margem")


def para_logits(valores, representacao):
    """Reconstrói logits para o softmax; a reconstrução é pós-processamento."""
    if representacao == "margem":
        return np.stack((-valores / 2.0, valores / 2.0), axis=-1)
    if representacao in ("original", "centrado"):
        return valores
    raise ValueError("Representação inválida")


def recortar_por_cliente(valores, limite, ordem):
    """Um único recorte L1 ou L2 por cliente, sobre todas as âncoras/classes.

    A fração recortada é um diagnóstico sensível da experiência, não um
    resultado seguro para publicar num protocolo real de privacidade.
    """
    x = np.asarray(valores, dtype=np.float64)
    if x.ndim not in (2, 3) or not np.isfinite(x).all():
        raise ValueError("Valores inválidos para recorte")
    if not np.isfinite(limite) or limite <= 0 or ordem not in (1, 2):
        raise ValueError("C deve ser positivo e a norma deve ser L1 ou L2")
    flat = x.reshape(x.shape[0], -1)
    normas = np.linalg.norm(flat, ord=ordem, axis=1)
    if not np.isfinite(normas).all():
        raise ValueError("Norma não finita")
    fatores = np.ones_like(normas)
    np.divide(limite, normas, out=fatores, where=normas > limite)
    fatores = np.minimum(fatores, 1.0)
    cortado = x * fatores.reshape((-1,) + (1,) * (x.ndim - 1))
    return cortado, normas > limite


def _log_delta_gaussiano(epsilon, sensibilidade, sigma):
    """Log do delta exato do mecanismo Gaussiano, em aritmética estável.

    delta = Phi(s/(2*sigma)-epsilon*sigma/s)
            - exp(epsilon)*Phi(-s/(2*sigma)-epsilon*sigma/s).
    """
    s = sensibilidade
    a = s / (2.0 * sigma) - epsilon * sigma / s
    b = -s / (2.0 * sigma) - epsilon * sigma / s
    log_a = float(log_ndtr(a))
    log_b = epsilon + float(log_ndtr(b))
    diferenca = log_b - log_a
    if diferenca >= 0:
        return -math.inf  # limite de precisão numérica em cauda extrema
    return log_a + math.log(-math.expm1(diferenca))


def sigma_gaussiano_analitico(epsilon, delta, sensibilidade):
    """Calibra o sigma mínimo aproximado por bisseção da equação analítica.

    A sensibilidade é L2. A validação numérica não substitui uma biblioteca
    de DP auditada, nem corrige problemas de precisão finita em produção.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon deve ser positivo e finito")
    if not np.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta gaussiano deve estar entre 0 e 1")
    if not np.isfinite(sensibilidade) or sensibilidade <= 0:
        raise ValueError("sensibilidade deve ser positiva e finita")
    alvo = math.log(delta)
    inferior, superior = 0.0, float(sensibilidade)
    for _ in range(200):
        if _log_delta_gaussiano(epsilon, sensibilidade, superior) <= alvo:
            break
        superior *= 2.0
    else:
        raise ValueError("Não foi possível calibrar sigma")
    for _ in range(100):
        meio = (inferior + superior) / 2.0
        if _log_delta_gaussiano(epsilon, sensibilidade, meio) > alvo:
            inferior = meio
        else:
            superior = meio
    return superior * (1.0 + 1e-12)  # arredondar levemente para o lado conservador


def parametros_ruido(mecanismo, limite, epsilon_rodada, delta_rodada):
    """Retorna ordem da norma, sensibilidade e escala/desvio por coordenada."""
    if not np.isfinite(limite) or limite <= 0:
        raise ValueError("C deve ser positivo e finito")
    if not np.isfinite(epsilon_rodada) or epsilon_rodada <= 0:
        raise ValueError("epsilon_rodada deve ser positivo e finito")
    sensibilidade = 2.0 * limite  # substituição arbitrária do vetor do cliente
    if mecanismo == "laplace":
        if delta_rodada != 0:
            raise ValueError("Laplace pura exige delta_rodada=0")
        return 1, sensibilidade, sensibilidade / epsilon_rodada
    if mecanismo == "gaussiano":
        sigma = sigma_gaussiano_analitico(epsilon_rodada, delta_rodada, sensibilidade)
        return 2, sensibilidade, sigma
    raise ValueError("Mecanismo deve ser laplace ou gaussiano")


def privatizar_simulado(logits, mecanismo, representacao, limite, epsilon_rodada,
                        delta_rodada, rng):
    """Devolve logits ruidosos e flag de clipping PARA TESTE LOCAL OFFLINE.

    rng tem semente reproduzível para investigação. Em produção, usar uma
    biblioteca auditada e aleatoriedade segura com contabilidade real.
    """
    ordem, _, escala = parametros_ruido(mecanismo, limite, epsilon_rodada, delta_rodada)
    valores = representar(logits, representacao)
    cortado, foi_recortado = recortar_por_cliente(valores, limite, ordem)
    if mecanismo == "laplace":
        ruido = rng.laplace(0.0, escala, size=cortado.shape)
    else:
        ruido = rng.normal(0.0, escala, size=cortado.shape)
    return para_logits(cortado + ruido, representacao), foi_recortado


def softmax_binario(logits):
    """Softmax de temperatura 1, por cliente/âncora, estável numericamente."""
    z = np.asarray(logits, dtype=np.float64)
    if z.shape[-1] != 2 or not np.isfinite(z).all():
        raise ValueError("Softmax espera logits binários finitos")
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
