from numbers import Number
from typing import Callable

import numpy as np
import pandas as pd


Vetor = np.array
Matriz = np.array
TabelaDeDados = pd.DataFrame
Modelo = Callable[[Matriz], Vetor] # f: X -> y
Preproc = Callable[[TabelaDeDados], Matriz]


def calcular_RMSE(previsões: Vetor, alvos: Vetor) -> Number:
    erros = previsões - alvos
    RMSE = np.sqrt(np.mean(erros**2))
    return RMSE