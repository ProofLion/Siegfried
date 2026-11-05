from collections import deque

import numpy as np

from models.grey_model import AbstractGreyModel
from models.measurement import Measurement

class GDF():
    def __init__(
        self,
        base_model: AbstractGreyModel,
        residual_model: AbstractGreyModel | None,
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
        x: Measurement
    ) -> None:
        self.history.append(x)

    def predict(
        self,
        steps_ahead: int,
        dt_seconds: float
    ) -> Measurement:
        if not self.ready:
            raise ValueError(f"GDF requires {self.window_size} measurements, but only has {len(self.history)}")
        
        self.base_model.fit( np.stack(self.history) )

        return self.base_model.predict(steps_ahead)
