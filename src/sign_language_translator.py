from contextlib import suppress
from typing import Literal

import cv2
import numpy as np
from src.constants import Key, Mode
from src.visuals import BOX_COLOUR, HAND_LANDMARK_STYLE, Colour
from src.dataclass import BoundingBox
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
import itertools


class SignLanguageTranslator:
    def __init__(
        self,
        max_num_hands: int = 4,
        model_complexity: Literal[0, 1] = 1,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.show_landmarks = True
        self.show_bounding_box = True
        self.padding = 30

        self.mode = Mode.FREEFORM
        self.mode_box_colour = Colour.CYAN
        self.mode_landmark_style = HAND_LANDMARK_STYLE[self.mode]

    def start(self) -> None:
        with suppress(KeyboardInterrupt):
            self.capture_video()

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode
        if mode != Mode.SELECT:
            self.mode_box_colour = BOX_COLOUR[mode]
            self.mode_landmark_style = HAND_LANDMARK_STYLE[mode]

    def capture_video(self) -> None:
        video_capture = cv2.VideoCapture(0)

        with self.hands as hands:
            while video_capture.isOpened():
                success, image = video_capture.read()
                if not success:
                    break

                # To improve performance, set as not writeable to pass by reference
                image.flags.writeable = False
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(
                        results.multi_hand_landmarks,
                        results.multi_handedness,
                        strict=False,
                    ):
                        bounding_box = self.get_bounding_box(
                            image=image, landmarks=hand_landmarks.landmark
                        )

                        self.draw_landmarks(image=image, hand_landmarks=hand_landmarks)
                        self.get_normalized_landmark_coordinates(
                            image=image, landmarks=hand_landmarks.landmark
                        )
                        self.draw_bounding_box(image=image, bounding_box=bounding_box)
                        self.draw_bounding_box_label(
                            image=image,
                            text=handedness.classification[0].label,
                            bounding_box=bounding_box,
                        )

                self.draw_metrics(
                    image=image,
                    num_hands=len(results.multi_hand_landmarks)
                    if results.multi_hand_landmarks
                    else 0,
                )

                cv2.imshow("Sign Language Alphabet Translator", image)
                match cv2.waitKey(5) & 0xFF:
                    case Key.M if self.mode != Mode.SELECT:
                        self.mode = Mode.SELECT
                    case Key.F if self.mode == Mode.SELECT:
                        self.switch_mode(Mode.FREEFORM)
                    case Key.D if self.mode == Mode.SELECT:
                        self.switch_mode(Mode.DATA_COLLECTION)
                    case Key.One:
                        self.show_landmarks ^= True
                    case Key.Two:
                        self.show_bounding_box ^= True
                    case Key.Esc:
                        break

        video_capture.release()
        cv2.destroyAllWindows()

    def get_landmark_coordinates(
        self, image: np.ndarray, landmarks: RepeatedCompositeFieldContainer
    ) -> np.ndarray:
        image_height, image_width, _ = image.shape

        coordinates: list[tuple[int, int]] = []
        for landmark in landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            point = (x, y)
            coordinates.append(point)

        return np.array(coordinates)

    def get_bounding_box(
        self, image: np.ndarray, landmarks: RepeatedCompositeFieldContainer
    ) -> BoundingBox:
        landmark_coordinates = self.get_landmark_coordinates(image, landmarks)
        x, y, width, height = cv2.boundingRect(landmark_coordinates)

        return BoundingBox(
            x=x - self.padding,
            y=y - self.padding,
            x2=width + x + self.padding,
            y2=height + y + self.padding,
        )

    def get_normalized_landmark_coordinates(
        self, image: np.ndarray, landmarks: RepeatedCompositeFieldContainer
    ) -> np.ndarray:
        landmark_coordinates = self.get_landmark_coordinates(image, landmarks)
        base_x, base_y = landmark_coordinates[0]

        relative_coordinates: list[tuple[int, int]] = [(0, 0)]
        for coordinate in landmark_coordinates[1:]:
            landmark_x, landmark_y = coordinate
            relative_point = (landmark_x - base_x, landmark_y - base_y)
            relative_coordinates.append(relative_point)

        # for r in relative_coordinates:
        #     x, y = r
        #     self.draw_text(
        #         image=image, text=" ", position=(x + 300, y + 500), font_scale=0.5
        #     )

        # One-dimensional
        flattened_coordinates = list(
            itertools.chain.from_iterable(relative_coordinates)
        )

        # Min-max normalization
        max_value = max(abs(coordinate) for coordinate in flattened_coordinates)
        normalized_coordinates = [
            coordinate / max_value for coordinate in flattened_coordinates
        ]

        return np.array(normalized_coordinates)

    def draw_bounding_box(self, image: np.ndarray, bounding_box: BoundingBox) -> None:
        if not self.show_bounding_box:
            return

        cv2.rectangle(
            img=image,
            pt1=(bounding_box.x, bounding_box.y),
            pt2=(bounding_box.x2, bounding_box.y2),
            color=self.mode_box_colour.value,
            thickness=2,
        )

    def draw_landmarks(
        self, image: np.ndarray, hand_landmarks: NormalizedLandmarkList
    ) -> None:
        if not self.show_landmarks:
            return

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=hand_landmarks,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mode_landmark_style,
        )

    def draw_text(
        self,
        image: np.ndarray,
        text: str,
        position: tuple[int, int],
        padding: int = 5,
        font: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 1.0,
        font_thickness: int = 1,
        text_colour: Colour = Colour.WHITE,
        background_colour: Colour = Colour.BLACK,
    ) -> None:
        x, y = position
        text_size, _ = cv2.getTextSize(
            text=text, fontFace=font, fontScale=font_scale, thickness=font_thickness
        )
        text_width, text_height = text_size

        cv2.rectangle(
            img=image,
            pt1=position,
            pt2=(
                x + text_width + (padding * 2),
                y + text_height + (padding * 2),
            ),
            color=background_colour.value,
            thickness=-1,
        )
        cv2.putText(
            img=image,
            text=text,
            org=(
                x + padding + 1,
                y + text_height + padding - 1,
            ),
            fontFace=font,
            fontScale=font_scale,
            color=text_colour.value,
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_bounding_box_label(
        self, image: np.ndarray, text: str, bounding_box: BoundingBox
    ) -> None:
        point_x = (
            bounding_box.x
            if self.show_bounding_box
            # Centre the label if no box
            else (bounding_box.x2 + bounding_box.x - (len(text) * 15)) // 2
        )
        point_y = bounding_box.y

        self.draw_text(
            image=image,
            text=text.upper(),
            position=(point_x - 1, point_y - 25),
            font_scale=0.65,
            font_thickness=1,
            text_colour=Colour.BLACK,
            background_colour=self.mode_box_colour,
        )

    def draw_metrics(self, image: np.ndarray, num_hands: int) -> None:
        messages: tuple[str, ...]
        if self.mode == Mode.SELECT:
            messages = (
                "Select Mode",
                "[F] Freeform",
                "[D] Data Collection",
                "[1] Toggle Landmarks",
                "[2] Toggle Bounding Box",
            )
        else:
            messages = (
                f"{self.mode} Mode",
                "[M] Change mode",
                f"Hands Detected: {num_hands}",
            )

        y_position = 25
        for y, text in enumerate(messages):
            self.draw_text(
                image=image,
                text=text,
                position=(25, y_position),
                font_scale=0.75 if y == 0 else 0.5,
            )
            y_position += 30 if y == 0 else 25
