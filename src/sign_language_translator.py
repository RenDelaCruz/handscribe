from contextlib import suppress
from typing import Literal

import cv2
import numpy as np
from constants import HAND_LANDMARK_STYLE, Colour
from dataclass import BoundingBox
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands


class SignLanguageTranslator:
    def __init__(
        self,
        max_num_hands: int = 4,
        model_complexity: Literal[0, 1] = 1,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.5,
        show_landmarks: bool = True,
        show_bounding_box: bool = True,
        bounding_box_padding: int = 30,
    ) -> None:
        self.hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.show_landmarks = show_landmarks
        self.show_bounding_box = show_bounding_box
        self.padding = bounding_box_padding

    def start(self) -> None:
        with suppress(KeyboardInterrupt):
            self.capture_video()

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
                        self.draw_bounding_box(image=image, bounding_box=bounding_box)
                        self.draw_bounding_box_label(
                            image=image,
                            text=handedness.classification[0].label,
                            bounding_box=bounding_box,
                        )

                # Draw metrics to screen
                messages = (
                    "Toggle Landmarks: Press 1",
                    "Toggle Bounding Box: Press 2",
                    f"Hands Detected: {len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0}",
                    "ESC to quit",
                )
                for y, text in enumerate(messages, start=1):
                    self.draw_text(
                        image=image,
                        text=text,
                        position=(25, y * 25),
                        font_scale=0.5,
                    )

                cv2.imshow("Sign Language AI Translator", image)
                match cv2.waitKey(5) & 0xFF:
                    case 49:  # Press 1
                        self.show_landmarks ^= True
                    case 50:  # Press 2
                        self.show_bounding_box ^= True
                    case 27:  # Press ESC
                        break

        video_capture.release()
        cv2.destroyAllWindows()

    def get_bounding_box(
        self, image: np.ndarray, landmarks: RepeatedCompositeFieldContainer
    ) -> BoundingBox:
        image_height, image_width, _ = image.shape

        landmark_points = np.empty((0, 2), int)
        for landmark in landmarks:
            x = int(landmark.x * image_width)
            y = int(landmark.y * image_height)
            point = [np.array((x, y))]
            landmark_points = np.append(landmark_points, point, axis=0)

        x, y, width, height = cv2.boundingRect(landmark_points)

        return BoundingBox(
            x=x - self.padding,
            y=y - self.padding,
            x2=width + x + self.padding,
            y2=height + y + self.padding,
        )

    def draw_bounding_box(self, image: np.ndarray, bounding_box: BoundingBox) -> None:
        if not self.show_bounding_box:
            return

        # cv2.circle(
        #     image,
        #     (
        #         (bounding_box.x + bounding_box.x2) // 2,
        #         (bounding_box.y + bounding_box.y2) // 2,
        #     ),
        #     max(
        #         bounding_box.y2 - bounding_box.y,
        #         bounding_box.x2 - bounding_box.x,
        #     )
        #     // 2 + self.padding,
        #     Colour.CYAN.value,
        #     2,
        # )
        # return

        cv2.rectangle(
            img=image,
            pt1=(bounding_box.x, bounding_box.y),
            pt2=(bounding_box.x2, bounding_box.y2),
            color=Colour.CYAN.value,
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
            landmark_drawing_spec=HAND_LANDMARK_STYLE,
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
                y + text_height + (padding * 2) - 1,
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
            background_colour=Colour.CYAN,
        )
