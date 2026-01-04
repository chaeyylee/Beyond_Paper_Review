import tensorflow as tf
from tensorflow.keras.models import load_model

MODEL_PATH = "mini_googlenet_cifar10.h5"
BATCH_SIZE = 64

def main():
    model = load_model(MODEL_PATH)

    print("📛 모델 출력 이름:", model.output_names)

    # CIFAR-10 로드
    (_, _), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()

    # ✅ 핵심 1: resize
    x_val = tf.image.resize(x_val, (224, 224))
    x_val = x_val / 255.0

    # one-hot
    y_val = tf.keras.utils.to_categorical(y_val, 10)

    # ✅ 핵심 2: 입력은 1개, 출력은 2개
    val_ds = tf.data.Dataset.from_tensor_slices(
        (x_val, (y_val, y_val))
    ).batch(BATCH_SIZE)

    # 평가
    results = model.evaluate(val_ds)

    print("\n📊 [평가 결과]")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

if __name__ == "__main__":
    main()
