from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class BoundingBox:
    x: int
    y: int
    x2: int
    y2: int
