from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from src.constants import Mode
from src.dataclass import BoundingBox
from src.sign_language_translator import SignLanguageTranslator
from src.visuals import BOX_COLOUR, HAND_LANDMARK_STYLE


class SwitchModeTests(TestCase):
    def setUp(self) -> None:
        self.translator = SignLanguageTranslator()

    def test__given_mode__switches_mode_and_styles(self) -> None:
        # given
        new_mode = Mode.DATA_COLLECTION

        # when
        self.translator.switch_mode(new_mode)

        # then
        self.assertEqual(self.translator.mode, new_mode)
        self.assertEqual(self.translator.mode_box_colour, BOX_COLOUR[new_mode])
        self.assertEqual(
            self.translator.mode_landmark_style, HAND_LANDMARK_STYLE[new_mode]
        )


@patch("cv2.rectangle")
class DrawBoundingBoxTests(TestCase):
    def setUp(self) -> None:
        self.translator = SignLanguageTranslator()
        self.image = Mock(shape=(100, 200, 3))

    def test__given_landmarks__returns_coordinate_points(
        self, mock_rectangle: MagicMock
    ) -> None:
        # given
        bounding_box = BoundingBox(x=30, y=30, x2=60, y2=60)

        # when
        self.translator.draw_bounding_box(image=self.image, bounding_box=bounding_box)

        # then
        mock_rectangle.assert_called_once_with(
            img=self.image,
            pt1=(30, 30),
            pt2=(60, 60),
            color=self.translator.mode_box_colour.value,
            thickness=2,
        )
