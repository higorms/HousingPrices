import numpy as np

from mlalgs.algs.regressor_linear import *


def test_treinar_regressor():
    np.random.seed(0)
    vetor_de_parâmetros_reais = np.array([7, 0, 1])

    características_aleatórias = np.random.normal(0, 1, (60, 2))

    X = np.concatenate([np.ones((60, 1)), características_aleatórias], axis=1)
    beta_real = vetor_de_parâmetros_reais
    ruídos = np.random.normal(0, 0.01, 60)

    y = X @ beta_real + ruídos

    alvos_e_características = np.concatenate([np.reshape(y, (60, 1)), características_aleatórias], axis=1)

    regressor, beta_estimado = treinar_regressor(alvos_e_características)
    previsões = regressor(características_aleatórias)

    assert np.isclose(previsões, y, atol=0.1).all()


