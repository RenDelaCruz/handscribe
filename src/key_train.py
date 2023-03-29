import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42

dataset = "models/data/key_coordinates.csv"
model_save_path = "models/key_classifier.hdf5"
tflite_save_path = "models/key_classifier.tflite"

NUM_CLASSES = 4  # 27  # 37

if __name__ == "__main__":
    # Read dataset
    x_dataset = np.loadtxt(
        dataset, delimiter=",", dtype="float32", usecols=list(range(1, (21 * 2) + 1))
    )
    y_dataset = np.loadtxt(dataset, delimiter=",", dtype="int32", usecols=(0))
    x_train, x_test, y_train, y_test = train_test_split(
        x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED
    )

    # Build model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((21 * 2,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(20, activation="relu"),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
        ]
    )

    input("Press [Enter] for model summary: ")

    model.summary()

    input("Press [Enter] to continue: ")

    # Model checkpoint callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        model_save_path, verbose=1, save_weights_only=False
    )
    # Callback for early stopping
    es_callback = tf.keras.callbacks.EarlyStopping(patience=50, verbose=1)
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

    input("Press [Enter] for model evaluation: ")

    # Model evaluation
    val_loss, val_acc = model.evaluate(x_test, y_test, batch_size=128)

    input("Press [Enter] to continue: ")

    # Loading the saved model
    model = tf.keras.models.load_model(model_save_path)

    input("Press [Enter] for inference test: ")

    # Inference test
    predict_result = model.predict(np.array([x_test[0]]))
    print(np.squeeze(predict_result))
    print(np.argmax(np.squeeze(predict_result)))

    input("Press [Enter] to continue: ")

    # Save as a model dedicated to inference
    model.save(model_save_path, include_optimizer=False)

    # Transform model (quantization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    with open(tflite_save_path, "wb") as f:
        f.write(tflite_quantized_model)

    # Additional inference test
    interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
    interpreter.allocate_tensors()

    # Get I / O tensor
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], np.array([x_test[0]]))

    interpreter.invoke()
    tflite_results = interpreter.get_tensor(output_details[0]["index"])

    print(np.squeeze(tflite_results))
    print(np.argmax(np.squeeze(tflite_results)))
