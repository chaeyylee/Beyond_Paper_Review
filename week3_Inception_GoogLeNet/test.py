import tensorflow as tf
from models.inception import build_googlenet
from utils.dataset import get_dataloader

model = build_googlenet(input_shape=(224,224,3), num_classes=10)
model.load_weights("saved_model.h5")

test_ds = get_dataloader(dataset='cifar10', train=False)
results = model.evaluate(test_ds)
print("Top-1 Accuracy:", results[1])
