from enum import Enum, IntEnum
from typing import Final


class Mode(Enum):
    def __str__(self) -> str:
        return self.value

    SELECT: Final[str] = "Select"
    FREEFORM: Final[str] = "Freeform"
    DATA_COLLECTION: Final[str] = "Data Collection"


class Key(IntEnum):
    # Based on ord() values
    Tab: Final[int] = 9
    Esc: Final[int] = 27
    Space: Final[int] = 32
    Zero: Final[int] = 48
    One: Final[int] = 49
    Two: Final[int] = 50
    Nine: Final[int] = 57
    A: Final[int] = 97
    D: Final[int] = 100
    F: Final[int] = 102
    Z: Final[int] = 122


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


CLASS_LABELS: Final[list[str]] = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "Space",
]


KEY_COORDINATES_CSV_PATH: Final[str] = "models/data/key_coordinates.csv"
