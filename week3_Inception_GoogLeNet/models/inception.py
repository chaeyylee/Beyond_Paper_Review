import tensorflow as tf
from tensorflow.keras import layers, Model

def inception_module(x, filters):
    f1, f3r, f3, f5r, f5, proj = filters

    path1 = layers.Conv2D(f1, (1,1), padding='same', activation='relu')(x)

    path2 = layers.Conv2D(f3r, (1, 1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(f3, (3, 3), padding='same', activation='relu')(path2)

    path3 = layers.Conv2D(f5r, (1, 1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(f5, (5, 5), padding='same', activation='relu')(path3)

    path4 = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    path4 = layers.Conv2D(proj, (1, 1), padding='same', activation='relu')(path4)

    return layers.Concatenate()([path1, path2, path3, path4])

def auxiliary_classifier(x, num_classes):
    x = layers.AveragePooling2D((5,5), strides=3)(x)
    x = layers.Conv2D(128, (1,1), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.7)(x)
    return layers.Dense(num_classes, activation='softmax')(x)

def build_googlenet(input_shape=(224, 224, 3), num_classes=1000):
    input_layer = layers.Input(shape=input_shape)

    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.Conv2D(64, (1, 1), activation='relu')(x)
    x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [64, 96, 128, 16, 32, 32])  # 3a
    x = inception_module(x, [128, 128, 192, 32, 96, 64])  # 3b
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [192, 96, 208, 16, 48, 64])  # 4a
    aux1 = auxiliary_classifier(x, num_classes)

    x = inception_module(x, [160, 112, 224, 24, 64, 64])  # 4b
    x = inception_module(x, [128, 128, 256, 24, 64, 64])  # 4c
    x = inception_module(x, [112, 144, 288, 32, 64, 64])  # 4d
    aux2 = auxiliary_classifier(x, num_classes)

    x = inception_module(x, [256, 160, 320, 32, 128, 128])  # 4e
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = inception_module(x, [256, 160, 320, 32, 128, 128])  # 5a
    x = inception_module(x, [384, 192, 384, 48, 128, 128])  # 5b

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs=input_layer, outputs=[output, aux1, aux2])
