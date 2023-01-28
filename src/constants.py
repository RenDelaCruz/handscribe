from enum import Enum
from typing import Final, TypeAlias

RGB: TypeAlias = tuple[int, int, int]


class Colour(Enum):
    BLACK: Final[RGB] = (0, 0, 0)
    WHITE: Final[RGB] = (255, 255, 255)
    GREEN: Final[RGB] = (0, 255, 0)
    YELLOW: Final[RGB] = (0, 255, 255)
    CYAN: Final[RGB] = (255, 255, 0)
