import numpy as np
import tensorflow as tf

from src.base.constants import CLASS_LABELS, TFLITE_SAVE_PATH


class KeyClassifier:
    def __init__(
        self,
        model_path: str = TFLITE_SAVE_PATH,
        num_threads: int = 1,
    ) -> None:
        self.interpreter = tf.lite.Interpreter(
            model_path=model_path, num_threads=num_threads
        )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def process(
        self,
        landmark_list: list[float],
    ) -> tuple[str, float]:
        input_details_tensor_index = self.input_details[0]["index"]
        self.interpreter.set_tensor(
            input_details_tensor_index, np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]["index"]

        result = self.interpreter.get_tensor(output_details_tensor_index)
        prediction = np.squeeze(result)
        prediction_index = int(np.argmax(prediction))
        return CLASS_LABELS[prediction_index], float(prediction[prediction_index])
