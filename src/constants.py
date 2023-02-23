from enum import Enum, IntEnum
from typing import Final, Literal, TypeAlias

from colour import Color as GradientPoint
from mediapipe.python.solutions.drawing_styles import DrawingSpec

BGR: TypeAlias = tuple[int, int, int]


class Colour(Enum):
    BLACK: Final[BGR] = (0, 0, 0)
    WHITE: Final[BGR] = (255, 255, 255)
    RED: Final[BGR] = (0, 0, 255)
    ORANGE: Final[BGR] = (0, 165, 255)
    YELLOW: Final[BGR] = (0, 255, 255)
    GREEN: Final[BGR] = (0, 255, 0)
    TEAL: Final[BGR] = (134, 166, 12)
    CYAN: Final[BGR] = (255, 245, 141)
    CERULEAN: Final[BGR] = (150, 131, 12)
    BLUE: Final[BGR] = (150, 87, 12)
    PURPLE: Final[BGR] = (150, 43, 117)
    MAGENTA: Final[BGR] = (150, 0, 153)


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


TEAL: Final[str] = "#00B37D"
MAGENTA: Final[str] = "#B52D75"
RGB_MAX: Literal[255] = 255

start_colour = GradientPoint(TEAL)
end_colour = GradientPoint(MAGENTA)
gradient = start_colour.range_to(end_colour, len(LandmarkPoint))
gradient_points = (
    (int(b * RGB_MAX), int(g * RGB_MAX), int(r * RGB_MAX))
    for r, g, b in (grad.rgb for grad in gradient)
)

HAND_LANDMARK_STYLE: Final[dict[LandmarkPoint, DrawingSpec]] = {
    point: DrawingSpec(
        color=next(gradient_points),
        thickness=3,
        circle_radius=4,
    )
    for point in LandmarkPoint
}
