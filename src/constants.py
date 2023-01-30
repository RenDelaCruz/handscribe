from enum import Enum, IntEnum
from typing import Final, TypeAlias

from mediapipe.python.solutions.drawing_styles import DrawingSpec

BGR: TypeAlias = tuple[int, int, int]


class Colour(Enum):
    BLACK: Final[BGR] = (0, 0, 0)
    WHITE: Final[BGR] = (255, 255, 255)
    RED: Final[BGR] = (0, 0, 255)
    ORANGE: Final[BGR] = (0, 165, 255)
    YELLOW: Final[BGR] = (0, 255, 255)
    GREEN: Final[BGR] = (0, 255, 0)
    MINT: Final[BGR] = (167, 255, 161)
    TEAL: Final[BGR] = (150, 186, 12)
    CYAN: Final[BGR] = (255, 245, 141)
    BLUE: Final[BGR] = (255, 0, 0)


class LandmarkPoint(IntEnum):
    WRIST: Final[int] = 0
    THUMB_CMC: Final[int] = 1
    THUMB_MCP: Final[int] = 2
    THUMB_IP: Final[int] = 3
    THUMB_TIP: Final[int] = 4
    INDEX_FINGER_MCP: Final[int] = 5
    INDEX_FINGER_PIP: Final[int] = 6
    INDEX_FINGER_DIP: Final[int] = 7
    INDEX_FINGER_TIP: Final[int] = 8
    MIDDLE_FINGER_MCP: Final[int] = 9
    MIDDLE_FINGER_PIP: Final[int] = 10
    MIDDLE_FINGER_DIP: Final[int] = 11
    MIDDLE_FINGER_TIP: Final[int] = 12
    RING_FINGER_MCP: Final[int] = 13
    RING_FINGER_PIP: Final[int] = 14
    RING_FINGER_DIP: Final[int] = 15
    RING_FINGER_TIP: Final[int] = 16
    PINKY_MCP: Final[int] = 17
    PINKY_PIP: Final[int] = 18
    PINKY_DIP: Final[int] = 19
    PINKY_TIP: Final[int] = 20


HAND_LANDMARK_STYLE: dict[LandmarkPoint, DrawingSpec] = {}
for point in LandmarkPoint:
    b, g, r = Colour.TEAL.value
    HAND_LANDMARK_STYLE[point] = DrawingSpec(
        color=(
            b,
            g - point.value * 6 if point.value < 5 else g - point.value * 11,
            r if g - point.value * 7 > 100 else point.value * 9,
        ),
        thickness=3,
        circle_radius=4,
    )
