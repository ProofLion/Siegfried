from abc import ABC
from dataclasses import dataclass

@dataclass
class Measurement(ABC):
    pass

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
