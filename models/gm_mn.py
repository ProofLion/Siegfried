from typing import Self

import numpy as np

from models.abstract_grey_model import AbstractGreyModel

class GM_MN(AbstractGreyModel):
    def __init__(self, m: int, n: int) -> None:
        self.m = m
        self.n = n

    def fit(self, X: np.ndarray) -> Self:
        self.X = X

        accumulated = np.cumsum(X, axis = 0)
        r = (accumulated[1:] + accumulated[:-1]) / 2
        Y = X[1:]
        B = np.column_stack([-r, np.ones_like(r)])
        
        self.a, self.b = np.linalg.lstsq(B, Y, rcond = None)[0]

        return self

    def predict(self, t: int | np.ndarray) -> float | np.ndarray:
        return (
            (self.X[0] - (self.b / self.a))
            * np.exp(-self.a * (t - 1))
            + self.b / self.a
        )

    def correct(self, t: int | np.ndarray) -> float | np.ndarray:
        return (
            (self.b - self.a * self.X[0])
            * np.exp(-self.a * (t - 1))
        )

