from collections import deque
from typing import Iterable

import numpy as np

from models.abstract_grey_model import AbstractGreyModel
from models.measurement import Measurement

class GDF():
    def __init__(
        self,
        base_model: AbstractGreyModel,
        residual_model: AbstractGreyModel,
        window_size: int,
    ) -> None:
        self.base_model     = base_model
        self.residual_model = residual_model
        self.window_size    = window_size
        self.history        = deque(maxlen = window_size)

    @property
    def ready(self) -> bool:
        return len(self.history) == self.window_size

    def update(
        self,
        x
    ) -> None:
        self.history.append(x)

    def predict(
        self,
        steps_ahead: int,
    ) -> float | np.ndarray:
        if not self.ready:
            raise ValueError(f"GDF requires {self.window_size} measurements, but only has {len(self.history)}")
        
        X = np.stack(self.history)
        n = X.shape[0]
        t = n + steps_ahead

        self.base_model.fit(X)

        actual = np.cumsum(X, axis = 0)
        fitted = np.asarray( self.base_model.predict( np.arange(1, n + 1) ) )

        residuals = actual - fitted

        self.residual_model.fit(residuals)

        def corrected_accumulated(t: int) -> float | np.ndarray:
            return self.base_model.predict(t) + self.residual_model.correct(t)

        return corrected_accumulated(t) - corrected_accumulated(t - 1)

