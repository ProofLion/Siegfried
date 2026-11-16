from abc import ABC
from typing import Self

import numpy as np

class AbstractGreyModel(ABC):
    def fit(self, X: np.ndarray) -> Self:
        return self

    def predict(self, t: int | np.ndarray) -> float | np.ndarray:
        return 0

    def correct(self, t: int | np.ndarray, *args, **kwargs) -> float | np.ndarray:
        return 0
