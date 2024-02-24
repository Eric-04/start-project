from data import generate_data
from preprocess import preprocess_data
import numpy as np
from model import model
import tensorflow as tf

if __name__ == "__main__":

    inputs, labels = generate_data()

    train_inputs, test_inputs, train_labels, test_labels = preprocess_data(inputs, labels, 0.8)


    # Compile model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train model
    model.fit(np.array(train_inputs), np.array(train_labels), epochs=5, batch_size=32, validation_split=0.2)

    # Evaluate model
    test_loss, test_acc = model.evaluate(np.array(test_inputs), np.array(test_labels))
    print('Test accuracy:', test_acc)


    # work on CNN


    # work on deep learning model


    # see the accuracy be above 90%