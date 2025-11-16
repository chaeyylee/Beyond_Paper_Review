from lenet1_model import build_lenet1
from model_stats import analyze_model_stats
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Preprocess
    x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')[..., np.newaxis] / 255.0
    x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')[..., np.newaxis] / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = build_lenet1()
    model.summary()

    # MAC 연산량 분석
    analyze_model_stats(model, model_name="LeNet-1")

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=15,
              validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\n[LeNet-1] Test Accuracy: {test_acc:.4f} / Error Rate: {(1 - test_acc):.4f}")

if __name__ == "__main__":
    main()
