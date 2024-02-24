from keras.models import Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Concatenate, AveragePooling2D, GlobalAveragePooling2D, Reshape

def dense_block(x, blocks, growth_rate):
    """A dense block."""
    for i in range(blocks):
        x = conv_block(x, growth_rate)
    return x

def conv_block(x, growth_rate):
    """A building block for a dense block."""
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, padding='same', kernel_initializer='he_normal')(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', kernel_initializer='he_normal')(x1)
    x = Concatenate()([x, x1])
    return x

def transition_block(x, reduction):
    """A transition block."""
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(x.shape[-1] * reduction), 1, padding='same', kernel_initializer='he_normal')(x)
    x = AveragePooling2D(2, strides=1, padding='same')(x)  # Adjusted pooling parameters
    return x

def DenseNet(input_shape, blocks=[6, 12, 24, 16], growth_rate=12, reduction=0.5, num_classes=None):
    """Instantiates the DenseNet architecture."""
    img_input = Input(shape=input_shape)

    # Reshape the input for compatibility with Conv2D layers
    x = Reshape((*input_shape, 1))(img_input)

    # Initial convolution layer
    x = Conv2D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(3, strides=2, padding='same')(x)

    # Dense blocks and transition blocks
    for i, block in enumerate(blocks):
        x = dense_block(x, block, growth_rate)
        if i != len(blocks) - 1:
            x = transition_block(x, reduction)

    # Final layers
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    if num_classes is not None:
        x = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(img_input, x)
    return model

# Example usage
input_shape = (28, 28)  # Adjusted input shape
num_classes = 9  # Adjusted number of classes
model = DenseNet(input_shape, num_classes=num_classes)
