import tensorflow as tf
from keras_flops import get_flops

def analyze_model_stats(model, model_name="LeNet"):
    # 모델 정보 출력
    model.summary()

    # FLOPs 계산
    flops = get_flops(model, batch_size=1)
    params = model.count_params()

    print(f"\n[{model_name}] 연산량 및 파라미터 수")
    print(f"Parameters: {params:,}")
    print(f"FLOPs (approx): {flops:,}")
