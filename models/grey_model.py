from typing import Iterable, Self

from models.measurement import Measurement

class AbstractGreyModel():
    def fit(self, X: Iterable[Measurement]) -> Self:
        return self

    def predict(self, steps: int = 1) -> Measurement:
        return Measurement()
        

class GreyModel(AbstractGreyModel):
    def __init__(self):
        pass

    def fit(
        self,
        X: Iterable[Measurement]
    ) -> Self:
        return self

    def predict(
        self,
        steps: int = 1
    ) -> Measurement:
        return Measurement()
