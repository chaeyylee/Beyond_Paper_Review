from tensorflow.keras import layers, models

def build_lenet1():
    model = models.Sequential()

    model.add(layers.Input(shape=(32, 32, 1)))  # padded MNIST

    # C1
    model.add(layers.Conv2D(filters=4, kernel_size=(5, 5), activation='tanh'))

    # S2
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Activation('tanh'))

    # C3
    model.add(layers.Conv2D(filters=12, kernel_size=(5, 5), activation='tanh'))

    # S4
    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=2))
    model.add(layers.Activation('tanh'))

    model.add(layers.Flatten())

    # Output: Dense(10), softmax
    model.add(layers.Dense(units=10, activation='softmax'))

    return model
