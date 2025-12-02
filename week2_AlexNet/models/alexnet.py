import torch
import torch.nn as nn
import torch.nn.functional as F

# 논문 Section 3.3 Local Response Normalization 구현
class LRN(nn.Module):
    def __init__(self, local_size=5, alpha=1e-4, beta=0.75, k=2.0):
        super(LRN, self).__init__()
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self,x):
        square = x.pow(2)
        # NCHW 기준으로 channel 방향으로 LRN 수행
        pad = (self.local_size - 1) // 2

        # channel 방향 padding
        extra_channels = F.pad(square, (0,0,0,0,pad,pad))
        scale = self.k

        for i in range(self.local_size):
            scale += self.alpha * extra_channels[:, i:i + x.size(1)] # channel-wise sum

        scale = scale.pow(self.beta)
        return x / scale

# 논문 Section 3 전체 구조 반영
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        # Section 3.1: ReLU 사용, Section 3.4: Overlapping pooling 적용, Section 3.3: LRN 적용
        self.features = nn.Sequential(
            # Conv1: 96 filters, 11x11, stride 4
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5),
            nn.MaxPool2d(kernel_size=3, stride=2), # overlapping pooling

            # Conv2: (GPU 분할 구조) 256 filters 5x5
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            LRN(local_size=5),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = torch.flatten(x,1) # batch dimension 유지
        x = self.classifier(x)
        return x