import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from torch.nn import functional as F

# Conv1 필터 시각화 함수
def visualize_first_conv_weights(model):
    # 첫 번째 Conv Layer weight 추출
    weights = model.features[0].weight.data.clone().cpu()

    # 정규화
    weights = (weights - weights.min()) / (weights.max() - weights.min())

    # 시각화 형태로 reshape (96 filters, 3 channel, 11x11)
    n_filters = weights.size(0)
    ncols = 8
    nrows = (n_filters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()

    for i in range(n_filters):
        img = weights[i].permute(1, 2, 0).numpy() # [C,H,W] -> [H,W,C]
        axes[i].imshow(img)
        axes[i].axis("off")
    plt.suptitle("Conv1 Filters (AlexNet)")
    plt.tight_layout()
    plt.show()

# 마지막 FC layer 기반 feature vector 추출
def get_feature_vectors(model, dataloader, device):
    model.eval()
    features, labels, images = [], [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model.features(inputs)
            outputs = torch.flatten(inputs)
            outputs = model.classifier[0:4](outputs) # FC1 -> ReLU -> FC2 -> ReLU
            features.append(outputs.cpu())
            labels.append(targets)
            images.append(inputs.cpu())

    return torch.cat(features), torch.cat(labels), torch.cat(images)

# 특정 쿼리 이미지와 가장 유사한 이미지 K개 시각화
def show_nearest_neighbors(query_idx, features, images, labels, k=6):
    query = features[query_idx]
    dists = torch.norm(features - query, dim=1)
    nearest = dists.topk(k+1, largest=False).indices # +1: 자기 자신 포함

    imgs = images[nearest].permute(0,2,3,1) # [B,C,H,W] -> [B,H,W,C]
    grid = make_grid(imgs.permute(0,3,1,2), nrow=k + 1)

    plt.figure(figsize=(15,3))
    plt.imshow(grid.permute(1,2,0))
    plt.axis("off")
    plt.title("Nearest Neighbors (query + top-{})".format(k))
    plt.show()