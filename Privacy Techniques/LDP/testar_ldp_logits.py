"""Testes de pesquisa: não demonstram segurança de uma implantação real."""
import math
import unittest
import numpy as np
from ldp_logits import (representar, para_logits, recortar_por_cliente,
    parametros_ruido, sigma_gaussiano_analitico, _log_delta_gaussiano,
    softmax_binario)
from experiencias_logits import carregar_rondas, FONTE_PADRAO


class TestarLDPLogits(unittest.TestCase):
    def test_representacoes_preservam_softmax_antes_do_recorte(self):
        rng = np.random.default_rng(9)
        z = rng.normal(size=(22, 32, 2)) * 10
        esperado = softmax_binario(z)
        for rep in ("original", "centrado", "margem"):
            obtido = softmax_binario(para_logits(representar(z, rep), rep))
            np.testing.assert_allclose(obtido, esperado, atol=1e-14)

    def test_recorte_conjunto_para_cada_cliente(self):
        rng = np.random.default_rng(5)
        for formato in ((22, 32, 2), (22, 32)):
            x = rng.normal(size=formato) * 100
            for ordem in (1, 2):
                cortado, flags = recortar_por_cliente(x, 7.0, ordem)
                normas = np.linalg.norm(cortado.reshape(22, -1), ord=ordem, axis=1)
                self.assertTrue(np.all(normas <= 7.0 + 1e-10))
                self.assertEqual(flags.shape, (22,))

    def test_sensibilidade_substituicao_limite_do_pior_caso(self):
        for ordem in (1, 2):
            x = np.ones((1, 32, 2))
            a, _ = recortar_por_cliente(x, 7.0, ordem)
            b, _ = recortar_por_cliente(-x, 7.0, ordem)
            distancia = np.linalg.norm((a - b).ravel(), ord=ordem)
            self.assertAlmostEqual(distancia, 14.0)

    def test_laplace_e_composicao_tres_rondas(self):
        epsilon_total, c = 4.0, 32.0
        eps_rodada = epsilon_total / 3.0
        ordem, sensibilidade, escala = parametros_ruido("laplace", c, eps_rodada, 0)
        self.assertEqual(ordem, 1)
        self.assertEqual(sensibilidade, 64.0)
        self.assertAlmostEqual(escala, 48.0)
        self.assertAlmostEqual(3 * eps_rodada, epsilon_total)

    def test_gaussiano_analitico_epsilon_menor_e_maior_que_um(self):
        for eps in (0.2, 0.9, 2.0, 5.0):
            delta, sensibilidade = 1e-5 / 3, 64.0
            sigma = sigma_gaussiano_analitico(eps, delta, sensibilidade)
            obtido = math.exp(_log_delta_gaussiano(eps, sensibilidade, sigma))
            self.assertLessEqual(obtido, delta * (1 + 1e-9))
            self.assertAlmostEqual(obtido / delta, 1.0, places=7)
            self.assertGreater(math.exp(_log_delta_gaussiano(eps, sensibilidade, sigma * .999)), delta)

    def test_recorte_sem_ruido_preserva_decisao_por_cliente(self):
        z = np.random.default_rng(2).normal(size=(22, 32, 2))
        for rep in ("original", "centrado", "margem"):
            vals = representar(z, rep)
            clipped, _ = recortar_por_cliente(vals, .5, 1)
            antes = np.argmax(softmax_binario(z), axis=-1)
            depois = np.argmax(softmax_binario(para_logits(clipped, rep)), axis=-1)
            np.testing.assert_array_equal(antes, depois)

    def test_entradas_invalidas(self):
        with self.assertRaises(ValueError): representar(np.full((22, 32, 2), np.nan), "original")
        with self.assertRaises(ValueError): parametros_ruido("laplace", 1, 0, 0)
        with self.assertRaises(ValueError): parametros_ruido("gaussiano", 1, 1, 0)
        with self.assertRaises(ValueError): recortar_por_cliente(np.ones((2, 3)), 0, 1)

    def test_tres_artefactos_reais_reproduzem_consenso(self):
        rondas = carregar_rondas(FONTE_PADRAO)
        self.assertEqual(len(rondas), 3)
        for ronda in rondas:
            self.assertEqual(ronda["logits"].shape, (22, 32, 2))
            np.testing.assert_allclose(ronda["pesos"].sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
