import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
from keras_preprocessing import image
import threading
from keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter


# --- Configuration ---
def train():
    IMG_SIZE = (28, 28)  # Adjust to match your actual image size
    BATCH_SIZE = 16
    DATASET_DIR = "drawings"  # Path to your image dataset folder
    VALIDATION_SPLIT = 0.2
    MIN_IMAGES = 26
    MIN_VALIDATION_IMAGES = 5  # Minimum images to keep for validation

    # Filter valid classes (non-empty folders)
    valid_classes = [
        cls for cls in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, cls)) and
        len(os.listdir(os.path.join(DATASET_DIR, cls))) > 0
    ]
    print(f"Using classes: {valid_classes}")

    # Count total images in valid classes
    total_images = 0
    for cls in valid_classes:
        cls_path = os.path.join(DATASET_DIR, cls)
        total_images += len(os.listdir(cls_path))
    print(f"Total images across classes: {total_images}")

    # Check if there are enough images for validation
    num_val_images = int(total_images * VALIDATION_SPLIT)
    if num_val_images < MIN_VALIDATION_IMAGES or total_images <= MIN_IMAGES:
        print(f"Warning: Not enough images ({num_val_images}) for validation. Skipping validation split.")
        use_validation = False
    else:
        use_validation = True

    # Set up ImageDataGenerator accordingly
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )

    if use_validation:
        datagen = ImageDataGenerator(validation_split=VALIDATION_SPLIT, rescale=1./255)
        train_gen = datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMG_SIZE,
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=valid_classes,
            subset='training',
            shuffle=True
        )
        val_gen = datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMG_SIZE,
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=valid_classes,
            subset='validation',
            shuffle=True
        )
    else:
        # No validation split, all data used for training
        train_gen = datagen.flow_from_directory(
            DATASET_DIR,
            target_size=IMG_SIZE,
            color_mode='grayscale',
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            classes=valid_classes,
            shuffle=True
        )
        val_gen = None  # No validation data

    print(Counter(train_gen.classes))

    # Compute class weights from training data only
    labels = train_gen.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights}")

    # Build model (same as before)
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(valid_classes), activation='softmax')
    ])

    model.compile(
        optimizer=optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
        ModelCheckpoint("models/best_model.keras", save_best_only=True)
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=35,
        class_weight=class_weights,
        callbacks=callbacks
    )

    model.save("models/image_classifier_grayscale.keras")
    print("Model saved as image_classifier_grayscale.keras")

def guess(IMG_PATH, model):
    IMG_SIZE = (28, 28)
    CLASS_NAMES = [
        "pizza", "tree", "house", "sun", "ball", "shoe", "cupcake", "fish",
        "ice cream", "cloud", "banana", "glasses", "key", "apple", "door", "bridge",
        "ladder", "cookie", "socks", "turtle", "drum", "robot", "ghost", "candle"
    ]

    img = image.load_img(IMG_PATH, target_size=IMG_SIZE, color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index]
    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


def guess_async(IMG_PATH, model, callback):
    def task():
        result = guess(IMG_PATH, model)
        callback(result)
    threading.Thread(target=task, daemon=True).start()
