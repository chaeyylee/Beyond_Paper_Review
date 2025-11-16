import matplotlib.pyplot as plt

def show_misclassified_samples(images, y_true, y_pred_classes, max_samples=20):
    """
    잘못 분류된 샘플 이미지와 예측 결과를 시각화
    """
    incorrect = (y_true != y_pred_classes)
    misclassified_images = images[incorrect]
    misclassified_true = y_true[incorrect]
    misclassified_pred = y_pred_classes[incorrect]

    num = min(max_samples, len(misclassified_images))
    if num == 0:
        print("잘못 분류된 이미지가 없습니다.")
        return

    plt.figure(figsize=(15, 2))
    for i in range(num):
        plt.subplot(1, num, i + 1)
        plt.imshow(misclassified_images[i].squeeze(), cmap='gray')
        plt.title(f"T:{misclassified_true[i]}\nP:{misclassified_pred[i]}")
        plt.axis('off')
    plt.suptitle("❌ 잘못 분류된 샘플 (T: 정답 / P: 예측)")
    plt.tight_layout()
    plt.show()
