# Beyond_Paper_Review
> AI 논문 분석 & 실습 스터디

---

## 📁 Week 1 : LeNet
- Yann LeCun의 고전적 CNN 구조 LeNet-5 논문을 기반으로 모델을 구현하고 MNIST 데이터에 대해 실험 수행
- 구현 : LeNet-1, LeNet-4, LeNet-5 구조, Discriminative Loss, 오답 시각화, Rejection 평가 등
- 🔗 [`week1_MNIST`](./week1_MNIST/)

---

## 📁 Week 2 : AlexNet
- 2012년 ILSVRC 우승 모델인 AlexNet 구조를 기반으로 CIFAR-10 데이터 실험 수행  
- 주요 구현 : ReLU, Local Response Normalization, Overlapping Pooling, Dropout, PCA 기반 Data Augmentation  
- 분석 및 시각화 : Conv1 필터 시각화, FC feature 기반 유사 이미지 검색  
- 🔗 [`week2_AlexNet`](./week2_AlexNet/)

---

## 📁 Week 3 : GoogLeNet (Inception)
- GoogLeNet 논문을 바탕으로 Inception 모듈과 Auxiliary Classifier가 포함된 전체 구조 구현
- 학습 시간 단축을 위해 Inception 3a~4a까지만 포함한 Mini-GoogLeNet도 함께 실험 진행
- CIFAR-10 데이터로 다중 출력 학습 수행, main/aux 분류기 정확도 비교 및 평가
- 🔗 [`week3_Inception_GoogLeNet`](./week3_Inception_GoogLeNet/)
