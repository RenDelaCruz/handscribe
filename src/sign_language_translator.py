from contextlib import suppress
from typing import Literal

import cv2
import numpy as np
from constants import Colour
from dataclass import BoundingBox
from google.protobuf.internal.containers import RepeatedCompositeFieldContainer
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands


class SignLanguageTranslator:
    def __init__(
        self,
        model_complexity: Literal[0, 1] = 0,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.5,
        show_landmarks: bool = True,
        show_bounding_box: bool = True,
        bounding_box_padding: int = 30,
    ) -> None:
        self.hands = mp_hands.Hands(
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
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference
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
                        self.draw_bounding_rectangle(
                            image=image, bounding_box=bounding_box
                        )
                        self.draw_label(
                            image=image,
                            text=handedness.classification[0].label,
                            bounding_box=bounding_box,
                        )

                # Flip the image horizontally for a selfie-view display
                cv2.imshow("Sign Language AI Translator", image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        video_capture.release()

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
            width=width + self.padding * 2,
            height=height + self.padding * 2,
        )

    def draw_bounding_rectangle(
        self, image: np.ndarray, bounding_box: BoundingBox
    ) -> None:
        if not self.show_bounding_box:
            return

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
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
        )

    def draw_label(
        self, image: np.ndarray, text: str, bounding_box: BoundingBox
    ) -> None:
        text_width = len(text) * 15
        point_x = (
            bounding_box.x
            if self.show_bounding_box
            # Centre the label if no box
            else (bounding_box.x2 + bounding_box.x - text_width) // 2
        )
        point_y = bounding_box.y

        cv2.rectangle(
            img=image,
            pt1=(point_x - 2, point_y - 2),
            pt2=(point_x + text_width, point_y - 30),
            color=Colour.BLACK.value,
            thickness=-1,
        )
        cv2.putText(
            img=image,
            text=text.upper(),
            org=(point_x, point_y - 7),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=Colour.WHITE.value,
            thickness=2,
        )
