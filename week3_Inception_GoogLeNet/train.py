import tensorflow as tf
from models.inception import build_googlenet
from utils.dataset import get_dataloader
from utils.config import *

model = build_googlenet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=10)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9),
    loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
    loss_weights=[1.0, 0.3, 0.3],
    metrics=['accuracy', 'accuracy', 'accuracy']
)

train_ds = get_dataloader(dataset='cifar10', train=True)
val_ds = get_dataloader(dataset='cifar10', train=False)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
