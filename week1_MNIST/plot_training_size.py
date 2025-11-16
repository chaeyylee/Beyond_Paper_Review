import matplotlib.pyplot as plt
import numpy as np

def plot_error_vs_trainingsize():
    # 훈련 데이터 크기
    train_sizes = [10_000, 20_000, 40_000, 60_000]

    # 논문 수치 기반
    train_error_no_distort = [0.5, 0.7, 0.75, 0.8]
    test_error_no_distort = [1.6, 1.2, 1.0, 0.8]
    test_error_with_distort = [0.6]  # 증강 시 (60k 기준)

    plt.figure(figsize=(6,4))
    plt.plot(np.array(train_sizes)//1000, test_error_no_distort, marker='s', label='Test error (no distortions)')
    plt.plot(np.array(train_sizes)//1000, train_error_no_distort, marker='o', label='Training error (no distortions)')
    plt.scatter([60], test_error_with_distort, facecolors='none', edgecolors='k', marker='s', s=100, label='Test error (with distortions)')
    plt.xlabel("Training Set Size (x1000)")
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate vs Training Set Size (Fig.6 Right)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_error_vs_trainingsize()
