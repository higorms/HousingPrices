import numpy as np
import numpy.linalg

from mlalgs.base import Matriz, Modelo, Vetor


def treinar_regressor(alvos_e_características: Matriz, termo_independente = True) -> Modelo:
    alvos = alvos_e_características[:, 0]
    características = alvos_e_características[:, 1:]

    número_de_observações, número_de_características = características.shape

    y = alvos
    if termo_independente:
        X = np.concatenate([np.ones((número_de_observações, 1)), características], axis=1)
    else:
        X = características

    beta_chapéu = vetor_de_parâmetros_ótimos = numpy.linalg.inv(X.T @ X) @ X.T @ y

    def regressor(características_de_teste: Matriz) -> Vetor:
        if termo_independente:
            X = np.concatenate([np.ones((número_de_observações, 1)), características_de_teste], axis=1)
        else:
            X = características_de_teste

        previsões = X @ beta_chapéu
        return previsões

    return regressor

