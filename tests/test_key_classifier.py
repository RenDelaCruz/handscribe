from unittest import TestCase

from src.key_classifier import KeyClassifier


class ProcessTests(TestCase):
    def setUp(self) -> None:
        self.key_classifier = KeyClassifier()

    def test__given_landmark_coordinates__returns_predicted_class_label_with_confidence(
        self,
    ) -> None:
        # given
        # Sample letter A
        landmark_list = [
            0.0,
            0.0,
            -0.3232758620689655,
            -0.11206896551724138,
            -0.6206896551724138,
            -0.36637931034482757,
            -0.7198275862068966,
            -0.6508620689655172,
            -0.5732758620689655,
            -0.8232758620689655,
            -0.4267241379310345,
            -0.7198275862068966,
            -0.43103448275862066,
            -0.9612068965517241,
            -0.3793103448275862,
            -0.7370689655172413,
            -0.35344827586206895,
            -0.5689655172413793,
            -0.21551724137931033,
            -0.7543103448275862,
            -0.22844827586206898,
            -1.0,
            -0.1939655172413793,
            -0.6982758620689655,
            -0.2025862068965517,
            -0.5431034482758621,
            -0.01293103448275862,
            -0.7413793103448276,
            -0.01293103448275862,
            -0.9612068965517241,
            -0.017241379310344827,
            -0.6681034482758621,
            -0.05172413793103448,
            -0.5129310344827587,
            0.1853448275862069,
            -0.6810344827586207,
            0.1853448275862069,
            -0.875,
            0.15086206896551724,
            -0.6810344827586207,
            0.10775862068965517,
            -0.5517241379310345,
        ]

        # when
        class_label, confidence = self.key_classifier.process(
            landmark_list=landmark_list
        )

        # then
        self.assertIsInstance(class_label, str)
        self.assertTrue(len(class_label), 1)
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 < confidence < 1)
