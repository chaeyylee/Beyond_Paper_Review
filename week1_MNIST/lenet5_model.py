from tensorflow.keras import layers, models

def build_lenet5():
    model = models.Sequential()

    # 입력 : 32x32 흑백 이미지
    model.add(layers.Input(shape=(32,32,1)))

    # C1: Conv2D (6 filters, 5x5), tanh
    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='tanh'))

    # S2: AveragePooling (2x2), stride 2, tanh
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Activation('tanh'))

    # C3: Conv2D (16 filters, 5x5), tanh
    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='tanh'))

    # S4: AveragePooling (2x2), stride 2, tanh
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Activation('tanh'))

    # C5: Conv2D (120 filters, 5x5), tanh
    model.add(layers.Conv2D(filters=120, kernel_size=(5, 5), activation='tanh'))

    # Flatten → F6: Dense(84), tanh
    model.add(layers.Flatten())
    model.add(layers.Dense(units=84, activation='tanh'))

    # Output layer: Dense(10), softmax
    model.add(layers.Dense(units=10, activation='softmax'))

    return model