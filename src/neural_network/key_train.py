from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src import _
from src.base.constants import (
    KEY_COORDINATES_DATASET_CSV_PATH,
    MODEL_SAVE_PATH,
    TFLITE_SAVE_PATH,
)

RANDOM_SEED = 42
NUM_POINTS = 21 * 2
NUM_CLASSES = 26


if __name__ == "__main__":
    # Read and split dataset
    x_dataset = np.loadtxt(
        KEY_COORDINATES_DATASET_CSV_PATH,
        delimiter=",",
        dtype=np.float32,
        usecols=list(range(1, NUM_POINTS + 1)),
    )
    y_dataset = np.loadtxt(
        KEY_COORDINATES_DATASET_CSV_PATH, delimiter=",", dtype=np.int32, usecols=(0)
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED
    )

    # Build model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((NUM_POINTS,)),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(60, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(80, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(60, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(40, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(30, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    # Model summary
    print(_("\nModel summary: "))
    model.summary()

    input(_("\nPress [Enter] to start model training: "))

    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        MODEL_SAVE_PATH, verbose=1, save_weights_only=False
    )

    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=60, verbose=1)

    # Model compilation
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # Model training
    model.fit(
        x_train,
        y_train,
        epochs=1000,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback, es_callback],
    )

    # Model evaluation
    print(_("\nModel evaluation: "))
    val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)

    # Loading the saved model
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    input(_("\nPress [Enter] for inference test: "))

    # Inference test
    predict_result = model.predict(np.array([x_test[0]]))
    print(np.squeeze(predict_result))
    print(np.argmax(np.squeeze(predict_result)))

    input(_("\nPress [Enter] to save model: "))

    # Save as a model dedicated to inference
    model.save(MODEL_SAVE_PATH, include_optimizer=False)

    # Transform model (quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    with Path(TFLITE_SAVE_PATH).open("wb") as f:
        f.write(tflite_quantized_model)

    # Additional inference test
    interpreter = tf.lite.Interpreter(model_path=TFLITE_SAVE_PATH)
    interpreter.allocate_tensors()

    # Get I / O tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], np.array([x_test[0]]))

    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]["index"])

    print(np.squeeze(tflite_results))
    print(np.argmax(np.squeeze(tflite_results)))

    print(_("\nProcess done!"))
