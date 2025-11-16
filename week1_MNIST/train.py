from lenet5_model import build_lenet5
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from visualize_misclassified import show_misclassified_samples

def discriminative_loss(y_true, y_pred):
    epsilon = 1e-7
    y_true = tf.cast(y_true, tf.float32)
    correct_log_prob = -tf.math.log(tf.reduce_sum(y_true * y_pred, axis=1) + epsilon)
    incorrect_probs = (1. - y_true) * y_pred
    incorrect_log_penalty = tf.reduce_mean(tf.math.log(incorrect_probs + epsilon), axis=1)
    loss = tf.reduce_mean(correct_log_prob - 0.5 * incorrect_log_penalty)
    return loss

def evaluate_with_rejection(y_pred, y_true, reject_threshold=0.2):
    top2 = np.sort(y_pred, axis=1)[:, -2:]
    confidence_gap = top2[:, 1] - top2[:, 0]
    rejected = confidence_gap < reject_threshold
    accepted = ~rejected

    total = len(y_true)
    accepted_count = np.sum(accepted)
    correct = np.sum((np.argmax(y_pred, axis=1) == y_true) & accepted)

    accuracy = correct / accepted_count if accepted_count > 0 else 0
    reject_rate = np.sum(rejected) / total

    print(f"\nRejection Threshold: {reject_threshold}")
    print(f"Accepted: {accepted_count}/{total} ({(1 - reject_rate):.2%})")
    print(f"Accuracy (Accepted Only): {accuracy:.4f}")
    print(f"Rejection Rate: {reject_rate:.2%}")

def main():
    # 1~4. 데이터 로딩 및 전처리
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_images = np.pad(train_images, ((0,0), (2,2), (2,2)), 'constant')[..., np.newaxis] / 255.0
    test_images = np.pad(test_images, ((0,0), (2,2), (2,2)), 'constant')[..., np.newaxis] / 255.0
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    # 5. 모델 구성
    model = build_lenet5()
    model.summary()

    # 6. 컴파일
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss=discriminative_loss,
                  metrics=['accuracy'])

    # 7. 데이터 증강
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1
    )
    datagen.fit(train_images)

    # 8. 학습
    model.fit(datagen.flow(train_images, train_labels, batch_size=64),
              steps_per_epoch=len(train_images) // 64,
              epochs=15,
              validation_data=(test_images, test_labels))

    # 9. 평가
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"\nTest Accuracy: {test_acc:.4f} / Error Rate: {(1 - test_acc):.4f}")

    # 10. 예측 결과 및 시각화
    y_pred = model.predict(test_images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    cm = confusion_matrix(y_true, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.show()

    show_misclassified_samples(test_images, y_true, y_pred_classes)

    # 11. Rejection 평가
    evaluate_with_rejection(y_pred, y_true, reject_threshold=0.2)
    evaluate_with_rejection(y_pred, y_true, reject_threshold=0.1)
    evaluate_with_rejection(y_pred, y_true, reject_threshold=0.05)

if __name__ == "__main__":
    main()
