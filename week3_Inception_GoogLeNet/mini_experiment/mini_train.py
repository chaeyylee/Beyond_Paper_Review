import tensorflow as tf
from mini_inception import build_mini_googlenet
from mini_dataset import get_dataloader
from mini_config import *

model = build_mini_googlenet(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), num_classes=NUM_CLASSES)

model.compile(
    optimizer="adam",
    loss={"main_output": "categorical_crossentropy","aux_output": "categorical_crossentropy"},
    loss_weights={"main_output": 1.0, "aux_output": 0.3},
    metrics={"main_output": "accuracy","aux_output": "accuracy"}
)

train_ds = get_dataloader(train=True)
val_ds = get_dataloader(train=False)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("mini_googlenet_cifar10.h5")
