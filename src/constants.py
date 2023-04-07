from enum import Enum, IntEnum
from typing import Final


class Mode(Enum):
    SELECT: Final[str] = "Select"
    FREEFORM: Final[str] = "Freeform"
    GAME: Final[str] = "Game"
    TIMED: Final[str] = "Timed"
    DATA_COLLECTION: Final[str] = "Data Collection"

    def __str__(self) -> str:
        return self.value


class Key(IntEnum):
    # Based on ord() values
    Tab: Final[int] = 9
    Esc: Final[int] = 27
    One: Final[int] = 49
    Two: Final[int] = 50
    A: Final[int] = 97
    D: Final[int] = 100
    F: Final[int] = 102
    G: Final[int] = 103
    T: Final[int] = 116
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
    "A",  # 0
    "B",  # 1
    "C",  # 2
    "D",  # 3
    "E",  # 4
    "F",  # 5
    "G",  # 6
    "H",  # 7
    "I",  # 8
    "J",  # 9
    "K",  # 10
    "L",  # 11
    "M",  # 12
    "N",  # 13
    "O",  # 14
    "P",  # 15
    "Q",  # 16
    "R",  # 17
    "S",  # 18
    "T",  # 19
    "U",  # 20
    "V",  # 21
    "W",  # 22
    "X",  # 23
    "Y",  # 24
    "Z",  # 25
]

KEY_COORDINATES_DATASET_CSV_PATH: Final[str] = "models/data/key_coordinates.csv"
MODEL_SAVE_PATH: Final[str] = "models/key_classifier.hdf5"
TFLITE_SAVE_PATH: Final[str] = "models/key_classifier.tflite"
WORDS_TXT_PATH: Final[str] = "models/data/words.txt"

SPELLING_MODE: Final[set[Mode]] = {Mode.GAME, Mode.TIMED}
