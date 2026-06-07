import os
import tensorflow as tf
import mlflow
import mlflow.tensorflow

DATASET_PATH = "mango_preprocessing"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

os.makedirs("outputs", exist_ok=True)

mlflow.tensorflow.autolog()

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224,2244,3)),
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128,3,activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation='relu'),

    tf.keras.layers.Dense(3,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

with mlflow.start_run():

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )

    model.save("outputs/mango_model.keras")

    mlflow.tensorflow.log_model(
        model, artifact_path="model"
    )

    print("Training selesai")