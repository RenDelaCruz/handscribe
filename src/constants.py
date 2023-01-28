from enum import Enum
from typing import Final, TypeAlias

BGR: TypeAlias = tuple[int, int, int]


class Colour(Enum):
    BLACK: Final[BGR] = (0, 0, 0)
    WHITE: Final[BGR] = (255, 255, 255)
    GREEN: Final[BGR] = (0, 255, 0)
    YELLOW: Final[BGR] = (0, 255, 255)
    CYAN: Final[BGR] = (255, 255, 0)
    ORANGE: Final[BGR] = (0, 165, 255)
