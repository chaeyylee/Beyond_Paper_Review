import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import numpy as np

# PCA 기반 조명 변형 클래스 (Lighting) 정의
class Lighting(object):
    def __init__(self, alphastd=0.1, eigval=None, eigvec=None):
        self.alphastd = alphastd
        self.eigval = eigval if eigval is not None else torch.Tensor([0.2175, 0.0188, 0.0045])
        self.eigvec = eigvec if eigvec is not None else torch.Tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ])

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = torch.normal(mean=0, std=self.alphastd, size=(3,))
        rgb = (self.eigvec @ (self.eigval * alpha)).numpy()
        img_np = np.asarray(img).astype(np.float32)
        for i in range(3):
            img_np[...,i] += rgb[i]
        img_np = np.clip(img_np, 0, 255)
        return transforms.functional.to_pil_image(img_np.astype(np.uint8))

# DataLoader 반환 함수
def get_loaders(batch_size=128, use_subset=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Train transform: crop + flip + lighting + normalize
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224), # 랜덤 크롭
        transforms.RandomHorizontalFlip(), # 좌우 반전
        Lighting(0.1), # 논문 Section 4.1 조명 변화
        transforms.ToTensor(),
        normalize,
    ])

    # Test transform: resize + center crop + normalize
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # 테스트 시 중앙 크롭
        transforms.ToTensor(),
        normalize,
    ])

    # CIFAR-10 dataset
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root='./data',train=False, download=True, transform=test_transform)

    if use_subset:
        train_set = Subset(train_set, range(5000))
        test_set = Subset(test_set, range(1000))

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader