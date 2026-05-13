import os

import numpy as np
import tensorflow as tf

from keras import layers
from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from hand_utils import load_class_names
from hand_utils import save_class_names


# ---------------- SETTINGS ----------------

DATA_DIR = "abc_data"

LABELS_PATH = "class_names.txt"

MODEL_PATH = "asl_best_model.keras"

MODEL_H5_PATH = "asl_best_model1.h5"

IMG_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 25

SEED = 42

VALIDATION_EVERY = 5

KEEP_EVERY = 2


# ---------------- LOAD DATASET ----------------

def collect_dataset():

    # Get all class names
    class_names = load_class_names(DATA_DIR)

    # Save class names in txt file
    save_class_names(class_names, LABELS_PATH)

    # Make dictionary
    class_to_index = {}

    for index, name in enumerate(class_names):
        class_to_index[name] = index

    train_paths = []
    train_labels = []

    val_paths = []
    val_labels = []

    # Go through every class folder
    for class_name in class_names:

        class_dir = os.path.join(DATA_DIR, class_name)

        image_names = []

        # Get all image names
        for image_name in os.listdir(class_dir):

            full_path = os.path.join(class_dir, image_name)

            if os.path.isfile(full_path):
                image_names.append(image_name)

        image_names.sort()

        # Keep every 2nd image
        image_names = image_names[::KEEP_EVERY]

        class_index = class_to_index[class_name]

        # Split images into train and validation
        for index, image_name in enumerate(image_names):

            image_path = os.path.join(class_dir, image_name)

            # Every 5th image goes to validation
            if index % VALIDATION_EVERY == 0:

                val_paths.append(image_path)

                val_labels.append(class_index)

            else:

                train_paths.append(image_path)

                train_labels.append(class_index)

    # Convert into numpy arrays
    train_paths = np.array(train_paths)

    train_labels = np.array(
        train_labels,
        dtype=np.int32
    )

    val_paths = np.array(val_paths)

    val_labels = np.array(
        val_labels,
        dtype=np.int32
    )

    # Shuffle training data
    rng = np.random.default_rng(SEED)

    order = rng.permutation(len(train_paths))

    train_paths = train_paths[order]

    train_labels = train_labels[order]

    print(f"Train images: {len(train_paths)}")

    print(f"Validation images: {len(val_paths)}")

    print(f"Classes: {class_names}")

    return (
        class_names,
        train_paths,
        train_labels,
        val_paths,
        val_labels
    )


# ---------------- CREATE DATASET ----------------

def build_dataset(paths, labels, class_count, training):

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    # Read image
    def load_image(path, label):

        image = tf.io.read_file(path)

        image = tf.image.decode_jpeg(
            image,
            channels=3
        )

        image = tf.image.resize(
            image,
            [IMG_SIZE, IMG_SIZE]
        )

        image = tf.cast(
            image,
            tf.float32
        )

        # Normalize image
        image = tf.keras.applications.mobilenet_v2.preprocess_input(
            image
        )

        # Convert label into one-hot vector
        label = tf.one_hot(
            label,
            depth=class_count
        )

        return image, label

    # Apply image loading
    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Shuffle only training data
    if training:

        dataset = dataset.shuffle(
            1000,
            seed=SEED
        )

    # Make batches
    dataset = dataset.batch(BATCH_SIZE)

    # Faster loading
    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset


# ---------------- CREATE MODEL ----------------

def build_model(class_count):

    # Data augmentation
    data_augmentation = tf.keras.Sequential(

        [
            layers.RandomFlip("horizontal"),

            layers.RandomRotation(0.08),

            layers.RandomZoom(0.1),

            layers.RandomTranslation(0.08, 0.08),

            layers.RandomContrast(0.1),
        ],

        name="augment",
    )

    # Load MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(

        input_shape=(IMG_SIZE, IMG_SIZE, 3),

        include_top=False,

        weights="imagenet",
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Input layer
    inputs = layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Apply augmentation
    x = data_augmentation(inputs)

    # Send into MobileNet
    x = base_model(
        x,
        training=False
    )

    # Convert feature maps into single vector
    x = layers.GlobalAveragePooling2D()(x)

    # Reduce overfitting
    x = layers.Dropout(0.4)(x)

    # Dense layer
    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    # Final output layer
    outputs = layers.Dense(
        class_count,
        activation="softmax"
    )(x)

    # Create model
    model = tf.keras.Model(
        inputs,
        outputs
    )

    # Compile model
    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=2e-4
        ),

        loss="categorical_crossentropy",

        metrics=["accuracy"],
    )

    return model


# ---------------- CLASS WEIGHTS ----------------

def make_class_weights(labels, class_count):

    counts = np.bincount(
        labels,
        minlength=class_count
    )

    total = counts.sum()

    class_weights = {}

    for index, count in enumerate(counts):

        if count > 0:

            class_weights[index] = (
                total / (class_count * count)
            )

    return class_weights


# ---------------- MAIN ----------------

def main():

    # Load dataset
    class_names, train_paths, train_labels, val_paths, val_labels = collect_dataset()

    # Create TensorFlow datasets
    train_ds = build_dataset(
        train_paths,
        train_labels,
        len(class_names),
        training=True
    )

    val_ds = build_dataset(
        val_paths,
        val_labels,
        len(class_names),
        training=False
    )

    # Calculate class weights
    class_weights = make_class_weights(
        train_labels,
        len(class_names)
    )

    # Build model
    model = build_model(
        len(class_names)
    )

    # Training callbacks
    callbacks = [

        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),

        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train model
    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=EPOCHS,

        callbacks=callbacks,

        class_weight=class_weights,
    )

    # Best accuracy
    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    print(
        f"Best validation accuracy: {best_val_accuracy:.4f}"
    )

    # Final evaluation
    val_loss, val_accuracy = model.evaluate(
        val_ds,
        verbose=2
    )

    print(
        f"Final validation loss: {val_loss:.4f}, accuracy: {val_accuracy:.4f}"
    )

    # Save final model
    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    model.save(MODEL_H5_PATH)

    print("Training complete")


# ---------------- START ----------------

if __name__ == "__main__":

    main()