import numpy as np
from keras import layers, models



# tf.keras.layers.Dropout(0.5)

# Define CNN model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    # layers.Reshape((28, 28, 1), input_shape=(28, 28)),  # Reshape input for convolutional layers
    # layers.Conv2D(3, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(3, (3, 3), activation='relu'),
    # layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(3, (3, 3), activation='relu'),
    # layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(9, activation='softmax') # 9 possible outputs
])