import csv
import itertools
from contextlib import suppress
import random
from typing import Literal

import cv2
import numpy as np
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands
from time import time
import datetime

from src.constants import (
    CLASS_LABELS,
    KEY_COORDINATES_DATASET_CSV_PATH,
    WORDS_TXT_PATH,
    Key,
    Mode,
)
from src.dataclass import BoundingBox, SuccessiveLetter
from src.key_classifier import KeyClassifier
from src.visuals import BOX_COLOUR, HAND_LANDMARK_STYLE, Colour

with open(WORDS_TXT_PATH, "r") as f:
    words = f.read().split("\n")


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

        # Data Collection mode
        self.pressed_key: str | None = None

        # Game and Alphabet mode
        self.points = 0
        self.current_word = ""
        self.letters_spelled = 0
        self.words_spelled = 0
        self.successive_letter: SuccessiveLetter | None = None

        # Timed mode
        self.max_time_seconds = 60
        self.start_timestamp = 0.0
        self.timer_seconds = self.max_time_seconds

        # Set to default Freeform modes
        self.mode = Mode.FREEFORM
        self.mode_box_colour = Colour.CYAN
        self.mode_landmark_style = HAND_LANDMARK_STYLE[self.mode]

        self.key_classifier = KeyClassifier()

    def start(self) -> None:
        with suppress(KeyboardInterrupt):
            self.capture_video()

    def switch_mode(self, mode: Mode) -> None:
        self.mode = mode
        match mode:
            case Mode.GAME | Mode.TIMED | Mode.ALPHABET:
                self.reset_to_next_word(
                    word="".join(CLASS_LABELS) if mode == Mode.ALPHABET else None
                )
                if mode in (Mode.TIMED, Mode.ALPHABET):
                    self.start_timestamp = int(time())
                    self.timer_seconds = self.max_time_seconds
            case Mode.FREEFORM | Mode.DATA_COLLECTION:
                self.mode_box_colour = BOX_COLOUR[mode]
                self.mode_landmark_style = HAND_LANDMARK_STYLE[mode]

                if mode == Mode.DATA_COLLECTION:
                    self.pressed_key = None

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
                        landmark_coordinates = self.get_landmark_coordinates(
                            image=image, landmarks=hand_landmarks.landmark
                        )
                        bounding_box = self.get_bounding_box(
                            landmark_coordinates=landmark_coordinates
                        )

                        # Classify hand landmarks to predicted class
                        normalized_coordinates = (
                            self.get_normalized_landmark_coordinates(
                                landmark_coordinates=landmark_coordinates,
                                flip=not handedness.classification[0].index,
                            )
                        )
                        class_label, confidence = self.key_classifier.process(
                            normalized_coordinates
                        )

                        self.draw_landmarks(image=image, hand_landmarks=hand_landmarks)
                        self.draw_bounding_box(image=image, bounding_box=bounding_box)
                        self.draw_bounding_box_label(
                            image=image,
                            text=handedness.classification[0].label
                            if self.mode == Mode.DATA_COLLECTION
                            else f"{class_label}  {confidence:0.2f}",
                            bounding_box=bounding_box,
                        )

                        if (
                            (self.mode == Mode.ALPHABET and self.letters_spelled < 26)
                            or (self.mode == Mode.TIMED and self.timer_seconds)
                            or self.mode == Mode.GAME
                        ):
                            self.spell_letter(
                                class_label=class_label, bounding_box=bounding_box
                            )

                self.draw_metrics(
                    image=image,
                    num_hands=len(results.multi_hand_landmarks)
                    if results.multi_hand_landmarks
                    else 0,
                )

                if self.mode in (Mode.GAME, Mode.TIMED, Mode.ALPHABET):
                    self.draw_game_words(image=image)
                    if self.mode != Mode.GAME:
                        self.draw_timer(image=image)

                cv2.imshow("Sign Language Alphabet Translator", image)
                match cv2.waitKey(5) & 0xFF:
                    case Key.Tab if self.mode != Mode.SELECT:
                        self.mode = Mode.SELECT
                    case Key.F if self.mode == Mode.SELECT:
                        self.switch_mode(Mode.FREEFORM)
                    case Key.A if self.mode == Mode.SELECT:
                        self.switch_mode(Mode.ALPHABET)
                    case Key.G if self.mode == Mode.SELECT:
                        self.switch_mode(Mode.GAME)
                    case Key.T if self.mode == Mode.SELECT:
                        self.switch_mode(Mode.TIMED)
                    case Key.D if self.mode == Mode.SELECT:
                        self.switch_mode(Mode.DATA_COLLECTION)
                    case Key.One:
                        self.show_landmarks ^= True
                    case Key.Two:
                        self.show_bounding_box ^= True
                    case key if (
                        Key.A <= key <= Key.Z
                    ) and self.mode == Mode.DATA_COLLECTION:
                        self.pressed_key = chr(key).upper()
                        key_id = key - Key.A
                        self.log_key_coordinates(
                            key_id=key_id, normalized_coordinates=normalized_coordinates
                        )
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

    def get_bounding_box(self, landmark_coordinates: np.ndarray) -> BoundingBox:
        x, y, width, height = cv2.boundingRect(landmark_coordinates)

        return BoundingBox(
            x=x - self.padding,
            y=y - self.padding,
            x2=width + x + self.padding,
            y2=height + y + self.padding,
        )

    def get_normalized_landmark_coordinates(
        self, landmark_coordinates: np.ndarray, flip: bool
    ) -> list[float]:
        base_x, base_y = landmark_coordinates[0]

        relative_coordinates: list[tuple[int, int]] = [(0, 0)]
        for coordinate in landmark_coordinates[1:]:
            landmark_x, landmark_y = coordinate
            relative_point = (
                # Flip coordinates for left hand
                (landmark_x - base_x) * (-1 if flip else 1),
                landmark_y - base_y,
            )
            relative_coordinates.append(relative_point)

        # for r in relative_coordinates:
        #     x, y = r
        #     self.draw_text(
        #         image=image,
        #         text=f"{(x,y)}",
        #         position=(x + 300, y + 500),
        #         font_scale=0.5
        #     )

        # One-dimensional
        flattened_coordinates = list(
            itertools.chain.from_iterable(relative_coordinates)
        )

        # Min-max normalization from [-1, 1]
        max_value = max(abs(coordinate) for coordinate in flattened_coordinates)
        normalized_coordinates = [
            coordinate / max_value for coordinate in flattened_coordinates
        ]

        return normalized_coordinates

    def log_key_coordinates(
        self, key_id: int, normalized_coordinates: list[float]
    ) -> None:
        with open(KEY_COORDINATES_DATASET_CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([key_id, *normalized_coordinates])

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
        thin: bool = False,
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
                x + text_width + (padding * 2) + 1,
                y + text_height + (padding * 2),
            )
            if thin
            else (
                x + text_width + int(padding * 2.5) + 1,
                y + text_height + (padding * 4) + 1,
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
            )
            if thin
            else (
                x + int(padding * 1.5) + 1,
                y + text_height + (padding * 2) - 1,
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
            else (bounding_box.x2 + bounding_box.x - (len(text) * 12)) // 2
        )
        point_y = bounding_box.y - (0 if self.show_bounding_box else 30)

        self.draw_text(
            image=image,
            text=text.upper(),
            position=(point_x - 1, point_y - 25),
            font_scale=0.65,
            font_thickness=1,
            text_colour=Colour.BLACK,
            background_colour=self.mode_box_colour,
            thin=True,
        )

    def reset_to_next_word(self, points: int = 0, word: str | None = None) -> None:
        self.current_word = word or self.get_random_word()
        self.letters_spelled = 0
        self.successive_letter = None

        if points:
            self.points += points
            self.words_spelled += 1
        else:
            self.points = 0
            self.words_spelled = 0

    def get_random_word(self) -> str:
        word = random.choice(words).upper()
        return word if word[0] != self.current_word[-1:] else self.get_random_word()

    def spell_letter(self, class_label: str, bounding_box: BoundingBox) -> None:
        word_length = len(self.current_word)
        if self.letters_spelled == word_length:
            self.reset_to_next_word(points=word_length)
        elif class_label == self.current_word[self.letters_spelled]:
            is_next_letter_the_same = (
                self.letters_spelled + 1 < word_length
                and self.current_word[self.letters_spelled]
                == self.current_word[self.letters_spelled + 1]
            )

            if not self.successive_letter:
                self.letters_spelled += 1

                if self.mode == Mode.ALPHABET:
                    self.points += 1
                elif is_next_letter_the_same:
                    width = bounding_box.x2 - bounding_box.x
                    centre_x = bounding_box.x + width // 2
                    self.successive_letter = SuccessiveLetter(
                        centre_x=centre_x, width=width
                    )
            elif (
                bounding_box.x2 > self.successive_letter.right_margin
                or bounding_box.x < self.successive_letter.left_margin
            ):
                self.letters_spelled += 1
                self.successive_letter = None

    def draw_game_words(self, image: np.ndarray) -> None:
        match self.mode:
            case Mode.ALPHABET:
                start = min(max(self.letters_spelled - 2, 0), 21)
                current_word = self.current_word[start : start + 5]
            case _:
                current_word = self.current_word
                start = 0

        x_position = 25
        font_scale = 3
        font_thickness = 8

        for index, letter in enumerate(current_word, start=start):
            text_size, _ = cv2.getTextSize(
                text=letter,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                thickness=font_thickness,
            )
            text_width, text_height = text_size
            self.draw_text(
                image=image,
                text=letter,
                position=(
                    x_position,
                    image.shape[0] - text_height - 25 * 2,
                ),
                font_scale=font_scale,
                font_thickness=font_thickness,
                text_colour=self.get_text_colour_based_on_countdown(index=index),
            )
            x_position += text_width + 10

    def draw_timer(self, image: np.ndarray) -> None:
        match self.mode:
            # Count down
            case Mode.TIMED:
                elapsed_time_seconds = int(time() - self.start_timestamp)
                self.timer_seconds = max(
                    self.max_time_seconds - elapsed_time_seconds, 0
                )
            # Count up
            case Mode.ALPHABET:
                if self.letters_spelled < 26:
                    self.timer_seconds = int(time() - self.start_timestamp)

        timer = datetime.timedelta(seconds=self.timer_seconds)
        self.draw_text(
            image=image,
            text=str(timer)[2:],
            position=(25, 135),
            font_scale=0.75,
            font_thickness=2,
            text_colour=self.get_text_colour_based_on_countdown(),
        )

    def get_text_colour_based_on_countdown(self, index: int | None = None) -> Colour:
        if self.mode in (Mode.ALPHABET, Mode.GAME):
            return (
                Colour.GREEN
                if (index is not None and self.letters_spelled > index)
                or (self.letters_spelled == 26 and self.mode == Mode.ALPHABET)
                else Colour.WHITE
            )

        if self.timer_seconds == 0 and (index is None or self.letters_spelled == 0):
            return Colour.RED

        return (
            (
                Colour.RED
                if self.timer_seconds == 0
                else Colour.ORANGE
                if self.timer_seconds < 10
                else Colour.YELLOW
                if self.timer_seconds < 20
                else Colour.GREEN
            )
            if index is not None and self.letters_spelled > index
            else Colour.WHITE
        )

    def draw_metrics(self, image: np.ndarray, num_hands: int) -> None:
        messages: tuple[str, ...]
        match self.mode:
            case Mode.SELECT:
                messages = (
                    "Select Mode",
                    "[F] Freeform",
                    "[A] Alphabet",
                    "[G] Game",
                    "[T] Timed",
                    "[D] Data Collection",
                    "[1] Toggle Landmarks",
                    "[2] Toggle Bounding Box",
                )
            case Mode.GAME | Mode.TIMED | Mode.ALPHABET:
                messages = (
                    f"{self.mode} Mode",
                    f"Current Letter: "
                    + f"{self.current_word[min(self.letters_spelled, 25)]}"
                    if self.mode == Mode.ALPHABET
                    else f"Words Spelled: {self.words_spelled}",
                    f"Points: {self.points}",
                )
            case Mode.DATA_COLLECTION:
                messages = (
                    f"Data Collection Mode",
                    f"Saved Key: {self.pressed_key}"
                    if self.pressed_key
                    else "Press a key to save data",
                )
            case _:
                messages = (
                    f"{self.mode} Mode",
                    f"Hands Detected: {num_hands}",
                )

        y_position = 25
        for y, text in enumerate(messages):
            self.draw_text(
                image=image,
                text=text,
                position=(25, y_position),
                font_scale=0.75 if y == 0 else 0.5,
                font_thickness=2 if y == 0 else 1,
            )
            y_position += 40 if y == 0 else 35
