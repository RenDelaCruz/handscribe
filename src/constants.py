from enum import IntEnum
from typing import Final


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
