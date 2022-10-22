from mlalgs.base import *


def test_calcular_RMSE():
    alvos = np.array([1, 0, 1])
    previsões = np.array([0.9, 0, 1])

    RMSE = calcular_RMSE(previsões, alvos)
    assert RMSE == 0.05773502691896256