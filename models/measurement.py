from abc import ABC
from dataclasses import dataclass
from typing import Iterable

@dataclass
class Measurement(ABC):
    def __add__(self, other):
        if not isinstance(other, Measurement):
            raise ValueError

        return Measurement()

    def __sub__(self, other):
        if not isinstance(other, Measurement):
            raise ValueError

        return Measurement()

@dataclass
class Position(Measurement):
    x: float
    y: float
    z: float

@dataclass
class OrbitalElements(Measurement):
    a:    float
    e_x:  float
    e_y:  float
    i_x:  float
    i_y:  float
    lmda: float
