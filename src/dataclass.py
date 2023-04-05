from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class BoundingBox:
    x: int
    y: int
    x2: int
    y2: int


@dataclass(frozen=True, kw_only=True)
class SuccessiveLetter:
    centre_x: int
    width: int

    @property
    def left_margin(self) -> int:
        return self.centre_x - self.width

    @property
    def right_margin(self) -> int:
        return self.centre_x + self.width
