import tensorflow as tf
from utils.transforms import preprocess_image, augment_image
from utils.config import BATCH_SIZE

def get_dataloader(dataset='cifar10', train=True):
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x,y = (x_train, y_train) if train else (x_test, y_test)
        y = tf.keras.utils.to_categorical(y, 10)

        ds = tf.data.Dataset.from_tensor_slices((x,y))
        ds = ds.map(augment_image if train else preprocess_image)
        ds = ds.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds
