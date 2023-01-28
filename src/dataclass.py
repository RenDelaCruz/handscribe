from dataclasses import dataclass


@dataclass(frozen=True)
class BoundingRectangle:
    x: int
    y: int
    width: int
    height: int
