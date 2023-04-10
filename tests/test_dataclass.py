from unittest import TestCase

from src.dataclass import SuccessiveLetter


class SuccessiveLetterTests(TestCase):
    def setUp(self) -> None:
        self.centre_x = 50
        self.width = 20
        self.successive_letter = SuccessiveLetter(centre_x=50, width=20)

    def test__returns_left_margin(self) -> None:
        # when
        result = self.successive_letter.left_margin

        # then
        self.assertEqual(result, self.centre_x - self.width)

    def test__returns_right_margin(self) -> None:
        # when
        result = self.successive_letter.right_margin

        # then
        self.assertEqual(result, self.centre_x + self.width)
