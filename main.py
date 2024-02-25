from data import generate_data
from preprocess import preprocess_data
import numpy as np
from model import DenseNet
from model_filter import create_base_model
import tensorflow as tf

if __name__ == "__main__":

    inputs, labels = generate_data()

    train_inputs, test_inputs, train_labels, test_labels = preprocess_data(inputs, labels, 0.8)

    image_size = len(test_inputs[0])
    input_shape = (image_size, image_size)  # Adjusted input shape
    num_classes = len(train_labels[0])  # Adjusted number of classes


    model = create_base_model(input_shape, num_classes)
    # model = DenseNet(input_shape, num_classes=num_classes)
    print(model.summary())

    model.compile(
        # optimizer=tf.keras.optimizers.Adadelta(learning_rate=1e-3, ),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6, amsgrad=True, ),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'])

    # Train model
    model.fit(np.array(train_inputs), np.array(train_labels), epochs=200, batch_size=len(train_inputs), validation_split=0.2)

    # Evaluate model
    test_loss, test_acc = model.evaluate(np.array(test_inputs), np.array(test_labels))
    print('Test accuracy:', test_acc)