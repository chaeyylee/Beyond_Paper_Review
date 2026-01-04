import tensorflow as tf
from mini_config import IMAGE_SIZE, BATCH_SIZE

def preprocess(image, label):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.squeeze(label, axis=-1)
    label = tf.one_hot(label, depth=10)

    return image, (label, label)

def get_dataloader(train=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x, y = (x_train, y_train) if train else (x_test, y_test)
    x, y = x[:10000], y[:10000]

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
