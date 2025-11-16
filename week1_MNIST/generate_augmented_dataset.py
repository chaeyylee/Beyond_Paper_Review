import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def generate_augmented_data(images, labels, augment_times=9, batch_size=1000, save_dir="augmented"):
    os.makedirs(save_dir, exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(images)

    X_aug = []
    y_aug = []

    print(f"Generating {augment_times}x augmented data...")
    for i in range(augment_times):
        print(f"  [Pass {i + 1}/{augment_times}]")
        for x_batch, y_batch in datagen.flow(images, labels, batch_size=batch_size, shuffle=False):
            X_aug.append(x_batch)
            y_aug.append(y_batch)
            if len(X_aug) * batch_size >= len(images):
                break

    X_aug = np.concatenate(X_aug)[:len(images) * augment_times]
    y_aug = np.concatenate(y_aug)[:len(images) * augment_times]

    print(f"Saving {X_aug.shape[0]} augmented samples to disk...")
    np.save(os.path.join(save_dir, "X_aug.npy"), X_aug)
    np.save(os.path.join(save_dir, "y_aug.npy"), y_aug)
    print("✅ Saved successfully.")


if __name__ == "__main__":
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant')[..., np.newaxis] / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    generate_augmented_data(x_train, y_train, augment_times=9)
