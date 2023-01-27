from contextlib import suppress
from typing import Literal

import cv2
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import hands as mp_hands


class SignLanguageTranslator:
    def __init__(
        self,
        model_complexity: Literal[0, 1] = 0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.hands = mp_hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

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
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=hand_landmarks,
                            connections=mp_hands.HAND_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                        )

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow("Sign Language AI Translator", cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        video_capture.release()
