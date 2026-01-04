# Week 3: GoogLeNet(Inception) 논문 구현

> **Going Deeper with Convolutions**  
> Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,  
> Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich (2014)

---

## 🧪 실험 목표

- GoogLeNet 논문에서 제안한 **Inception 구조의 설계 의도와 동작 방식**을 이해하고 직접 구현
- 병렬 합성곱 구조, 1×1 Convolution 기반 차원 축소, Auxiliary Classifier의 역할을 실험을 통해 확인
- 논문 구조를 최대한 반영한 기본 GoogLeNet 구현 후, 학습 시간 단축을 위한 Mini‑GoogLeNet 구조를 추가 실험
- CIFAR‑10 데이터셋을 사용하여 다중 출력(main / auxiliary) 학습 및 성능 비교 수행

---

## 📁 파일 구성

| 파일명          | 설명                                   |
|----------------|----------------------------------------|
| `inception.py` | GoogLeNet 전체 구조 정의               |
| `config.py`    | 전체 모델용 하이퍼파라미터 설정        |
| `dataset.py`   | CIFAR-10 데이터 로딩 함수 정의         |
| `transforms.py`| 데이터 전처리 및 증강 파이프라인 정의   |
| `train.py`     | GoogLeNet 학습 스크립트                |
| `test.py`      | 학습된 모델 평가 스크립트              |

---

## ✅ 주요 구현 내용

- [x] Inception 모듈 병렬 구조 구현 (1×1, 3×3, 5×5, Pooling branch)
- [x] 1×1 Convolution을 이용한 채널 차원 축소 구조 재현
- [x] Auxiliary Classifier 포함한 다중 출력 모델 구성
- [x] GoogLeNet 기본 구조 구현 후 Mini‑GoogLeNet 구조 추가 실험
- [x] main / auxiliary 출력 분리 학습 및 평가
- [x] CIFAR‑10 기준 학습 결과 비교 분석

---

## ▶️ 실행 방법

```bash
# GoogLeNet 학습 및 평가
python train.py
python test.py

# Mini-GoogLeNet 학습 및 평가
python mini_train.py
python mini_test.py

⚠️ CPU 환경에서는 학습 시간이 길어 Inception 3a~4a까지만 포함한 Mini 구조 사용
```

---

## 📊 실험 결과 요약

Mini‑GoogLeNet 구조를 CIFAR‑10 데이터셋 10,000개 기준으로 3 Epoch 학습한 결과는 다음과 같다.

| 항목 | 결과 |
|-----|-----|
| Epoch 수 | 3 |
| Train Main Accuracy | 27.78% |
| Validation Main Accuracy | 31.36% |
| Validation Auxiliary Accuracy | 38.77% |
| Validation Loss | 2.3439 |
| Overfitting 여부 | 없음 (train/validation 성능 유사) |

### Epoch 3 기준 주요 지표
- `main_output_accuracy` : **0.2778**
- `val_main_output_accuracy` : **0.3136**
- `val_aux_output_accuracy` : **0.3877**
- `val_loss` : **2.3439**

---

## 🔍 결과 해석

### 1. Auxiliary Classifier(다중 출력 구조)의 효과

- Auxiliary Classifier의 정확도(38.77%)가 Main Classifier(31.36%)보다 높게 나타남
- 이는 **중간 단계 feature map에서는 비교적 분류가 용이**한 반면, **더 깊은 layer에서는 학습이 아직 충분히 진행되지 않았음**을 의미
- 논문에서 제안한 것처럼 Auxiliary Classifier가 **gradient 소실을 완화하고 학습을 안정화하는 역할**을 수행하고 있음을 실험적으로 확인

### 2. 학습 안정성 분석

- Epoch 증가에 따라 train loss와 validation loss가 함께 감소
- validation accuracy가 지속적으로 상승하는 추세를 보임
- train/validation 성능 차이가 크지 않아 **오버피팅 현상은 관찰되지 않음**

### 3. Mini‑GoogLeNet(구조 축소)의 한계와 의미

- 원 논문 대비 Inception 모듈 수(3a~4a)와 데이터 규모를 크게 축소한 구조임
- 그럼에도 불구하고 랜덤 분류 기준(10%) 대비 **의미 있는 성능 확보**
- Inception 구조의 **병렬 합성곱과 1×1 Convolution 기반 표현 학습 효율성**을 소규모 실험에서도 확인할 수 있었음

> 본 실험은 GoogLeNet의 핵심 아이디어를 빠르게 검증하는 데 목적이 있으며,  
> 충분한 Epoch, 데이터 규모 확대 시 성능 향상이 기대됨

---

## 📚 참고 자료

- 📄 [논문 원문](https://arxiv.org/abs/1409.4842)
- 📘 [TensorFlow 공식 Inception 모델 설명](https://www.tensorflow.org/tutorials/images/cnn#training_a_cnn_on_cifar10)
- 📦 [CIFAR-10 데이터셋 설명](https://www.cs.tronto.edu/~kriz/cifar.html)
