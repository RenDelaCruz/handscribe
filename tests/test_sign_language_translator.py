from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from src.constants import Mode
from src.dataclass import BoundingBox
from src.sign_language_translator import SignLanguageTranslator
from src.visuals import BOX_COLOUR, HAND_LANDMARK_STYLE


@patch("src.sign_language_translator.SignLanguageTranslator.capture_video")
class StartTests(TestCase):
    def setUp(self) -> None:
        self.translator = SignLanguageTranslator()

    def test__starts_main_loop(self, mock_capture_video: MagicMock) -> None:
        # given
        mock_capture_video.side_effect = KeyboardInterrupt

        # when
        self.translator.start()

        # then
        mock_capture_video.assert_called_once_with()


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

    def test__given_select_mode__switches_mode_but_retains_old_styles(self) -> None:
        # given
        old_mode = self.translator.mode
        new_mode = Mode.SELECT

        # when
        self.translator.switch_mode(new_mode)

        # then
        self.assertEqual(self.translator.mode, new_mode)
        self.assertEqual(self.translator.mode_box_colour, BOX_COLOUR[old_mode])
        self.assertEqual(
            self.translator.mode_landmark_style, HAND_LANDMARK_STYLE[old_mode]
        )


class GetLandmarkCoordinatesTests(TestCase):
    def setUp(self) -> None:
        self.translator = SignLanguageTranslator()

    def test__returns_landmark_coordinates_in_relation_to_image(self) -> None:
        # given
        height, width = 720, 1280
        image = Mock(shape=(height, width, 3))
        landmarks = MagicMock()
        landmarks.__iter__.return_value = [
            Mock(x=x, y=y) for x, y in [(i / 22, i / 22) for i in range(1, 22)]
        ]

        # when
        result = self.translator.get_landmark_coordinates(
            image=image, landmarks=landmarks
        )

        # then
        self.assertTrue(
            np.array_equal(
                result,
                np.array(
                    [(int(width * i / 22), int(height * i / 22)) for i in range(1, 22)]
                ),
            )
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
