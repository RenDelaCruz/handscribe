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


FINGER_GROUP_COLOURS: Final[dict[tuple[LandmarkPoint, ...], Colour]] = {
    (
        LandmarkPoint.WRIST,
        LandmarkPoint.THUMB_CMC,
        LandmarkPoint.THUMB_MCP,
        LandmarkPoint.THUMB_IP,
        LandmarkPoint.THUMB_TIP,
    ): Colour.TEAL,
    (
        LandmarkPoint.INDEX_FINGER_MCP,
        LandmarkPoint.INDEX_FINGER_PIP,
        LandmarkPoint.INDEX_FINGER_DIP,
        LandmarkPoint.INDEX_FINGER_TIP,
    ): Colour.CERULEAN,
    (
        LandmarkPoint.MIDDLE_FINGER_MCP,
        LandmarkPoint.MIDDLE_FINGER_PIP,
        LandmarkPoint.MIDDLE_FINGER_DIP,
        LandmarkPoint.MIDDLE_FINGER_TIP,
    ): Colour.BLUE,
    (
        LandmarkPoint.RING_FINGER_MCP,
        LandmarkPoint.RING_FINGER_PIP,
        LandmarkPoint.RING_FINGER_DIP,
        LandmarkPoint.RING_FINGER_TIP,
    ): Colour.PURPLE,
    (
        LandmarkPoint.PINKY_MCP,
        LandmarkPoint.PINKY_PIP,
        LandmarkPoint.PINKY_DIP,
        LandmarkPoint.PINKY_TIP,
    ): Colour.MAGENTA,
}

HAND_LANDMARK_STYLE: dict[LandmarkPoint, DrawingSpec] = {}
for group, colour in FINGER_GROUP_COLOURS.items():
    for finger in group:
        b, g, r = colour.value
        HAND_LANDMARK_STYLE[finger] = DrawingSpec(
            color=(
                b,
                g - finger * 6
                if finger < LandmarkPoint.INDEX_FINGER_MCP
                else g - 10 * ((finger - 1) % 4),
                r
                if finger < LandmarkPoint.RING_FINGER_MCP
                else r + 10 * ((finger - 1) % 4),
            ),
            thickness=3,
            circle_radius=4,
        )
