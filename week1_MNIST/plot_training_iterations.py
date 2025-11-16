import matplotlib.pyplot as plt
import numpy as np

def plot_error_vs_iterations():
    # 학습 반복 횟수 (Epoch 또는 Pass 기준)
    iterations = np.arange(1, 22, 2)

    # 논문 기반 추정 수치
    training_error = [5.0, 2.8, 1.8, 1.2, 1.0, 0.9, 0.8, 0.75, 0.72, 0.71, 0.70]
    test_error =     [5.2, 3.5, 2.1, 1.7, 1.5, 1.4, 1.35, 1.33, 1.32, 1.32, 1.30]

    plt.figure(figsize=(6,4))
    plt.plot(iterations, training_error, marker='o', label='Training')
    plt.plot(iterations, test_error, marker='s', label='Test')
    plt.xlabel("Training set Iterations")
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate vs Training Iterations (Fig.6 Left)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_error_vs_iterations()
