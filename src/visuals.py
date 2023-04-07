from __future__ import annotations

from enum import Enum
from typing import Final, Iterator, Literal, TypeAlias

from colour import Color, color_scale
from mediapipe.python.solutions.drawing_styles import DrawingSpec

from src.constants import LandmarkPoint, Mode

BGR: TypeAlias = tuple[int, int, int]
RGB: TypeAlias = tuple[float, float, float]
RGB_MAX: Literal[255] = 255


class Colour(Enum):
    BLACK: Final[BGR] = (0, 0, 0)
    WHITE: Final[BGR] = (255, 255, 255)
    RED: Final[BGR] = (77, 26, 250)
    ORANGE: Final[BGR] = (0, 165, 255)
    YELLOW: Final[BGR] = (44, 230, 230)
    GREEN: Final[BGR] = (112, 247, 32)
    TEAL: Final[BGR] = (96, 138, 18)
    CYAN: Final[BGR] = (255, 245, 141)
    CERULEAN: Final[BGR] = (150, 131, 12)
    BLUE: Final[BGR] = (150, 87, 12)
    PURPLE: Final[BGR] = (150, 43, 117)
    MAGENTA: Final[BGR] = (89, 0, 153)

    @property
    def rgb(self) -> RGB:
        b, g, r = self.value
        return r / RGB_MAX, g / RGB_MAX, b / RGB_MAX


class GradientPoint(Color):
    def range_to(self, value: GradientPoint, steps: int) -> Iterator[GradientPoint]:
        for hsl in color_scale(self._hsl, Color(value).hsl, steps - 1):
            yield GradientPoint(hsl=hsl)

    @property
    def bgr(self) -> BGR:
        r, g, b = self.rgb
        return int(b * RGB_MAX), int(g * RGB_MAX), int(r * RGB_MAX)


BOX_COLOUR: Final[dict[Mode, Colour]] = {}
HAND_LANDMARK_STYLE: Final[dict[Mode, dict[LandmarkPoint, DrawingSpec]]] = {}

for mode, main_colour, start_colour, end_colour in (
    (Mode.FREEFORM, Colour.CYAN, Colour.TEAL, Colour.MAGENTA),
    (Mode.DATA_COLLECTION, Colour.YELLOW, Colour.ORANGE, Colour.TEAL),
):
    start = GradientPoint(rgb=start_colour.rgb)
    end = GradientPoint(rgb=end_colour.rgb)
    gradient = start.range_to(end, len(LandmarkPoint))
    gradient_points = (grad.bgr for grad in gradient)

    BOX_COLOUR[mode] = main_colour
    HAND_LANDMARK_STYLE[mode] = {
        point: DrawingSpec(
            color=next(gradient_points),
            thickness=2,
            circle_radius=2,
        )
        for point in LandmarkPoint
    }
