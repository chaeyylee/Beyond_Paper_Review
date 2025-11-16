import matplotlib.pyplot as plt

def show_sample_images(images, labels, num=10):
    plt.figure(figsize=(10, 1))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()