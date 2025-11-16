import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np

def build_lenet4():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32, 32, 1)),
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation='tanh'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Activation('tanh'),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation='tanh'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='tanh'),
        tf.keras.layers.Dense(84, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant') / 255.0
x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant') / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 앙상블 학습
models = []
for seed in [1, 2, 3]:
    tf.keras.utils.set_random_seed(seed)
    model = build_lenet4()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(x_train, y_train_cat, batch_size=64, epochs=5, verbose=1)
    models.append(model)

# 예측 앙상블
preds = np.stack([m.predict(x_test) for m in models], axis=0)
avg_preds = np.mean(preds, axis=0)
y_pred_classes = np.argmax(avg_preds, axis=1)
y_true = np.argmax(y_test_cat, axis=1)

# 정확도 출력
acc = np.mean(y_pred_classes == y_true)
print(f"Boosted LeNet-4 Accuracy: {acc:.4f} / Error Rate: {1 - acc:.4f}")
