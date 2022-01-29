from keras.models import Model
from keras.layers import (
    Input,
    MaxPooling2D,
    Conv2D,
    BatchNormalization,
    ReLU,
    Add,
    GlobalAveragePooling2D,
    Dense,
)
from tensorflow.keras.utils import plot_model


# RESNET Block
def resblock(inputX, block_num, filters, downsample_block=True):

    
    if block_num == 0 and downsample_block:
        x = Conv2D(filters=filters, kernel_size=1, strides=2, padding="same")(inputX)
    else:
        x = Conv2D(filters=filters, kernel_size=1, padding="same")(inputX)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters=filters * 4, kernel_size=1, padding="same")(x)
    x = BatchNormalization()(x)

    if block_num == 0:
        if downsample_block:
            inputX = Conv2D(
                filters=filters * 4, kernel_size=1, strides=2, padding="same"
            )(inputX)

        else:
            inputX = Conv2D(filters=filters * 4, kernel_size=1, padding="same")(inputX)

        inputX = BatchNormalization()(inputX)

    x = Add()([inputX, x])
    x = ReLU()(x)

    return x


def RESNET50():
    # Input
    inputX = Input(shape=(224, 224, 3))

    # First Downsample
    # output_size 112x112
    x = Conv2D(filters=64, kernel_size=7, strides=2, padding="same")(inputX)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=7, strides=2, padding="same")(x)

    # Block 1
    # output_size 56x56
    for i in range(3):
        x = resblock(x, i, filters=64, downsample_block=False)

    # Block 2
    # output_size 28x28
    for i in range(4):
        x = resblock(x, i, filters=128)

    # Block 3
    # output_size 14x14
    for i in range(6):
        x = resblock(x, i, filters=256)

    # Block 4
    # output_size 7x7
    for i in range(3):
        x = resblock(x, i, filters=512)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=1000, activation="softmax")(x)

    model = Model(inputs=inputX, outputs=x, name="ResNet50")

    return model


if __name__ == "__main__":
    model = RESNET50()
    model.summary()
    plot_model(model, show_shapes=True, to_file="RESNET50_shape.png")
    plot_model(model, show_shapes=False, to_file="RESNET50.png")
