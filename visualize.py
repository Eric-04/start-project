import matplotlib.pyplot as plt
import numpy as np


def visualize_metrics(history):
    if not history:
        return print("Must provide a history to visualize")
    
    accuracy = history['categorical_accuracy']
    val_accuracy = history['val_categorical_accuracy']


    x = np.arange(1, len(accuracy) + 1)

    plt.plot(x, accuracy, label='categorical_accuracy', color='blue')
    plt.plot(x, val_accuracy, label = 'val_categorical_accuracy', color='red')
    plt.ylabel("Accuracy Value")
    plt.title("Training Accuracy Over Epoch")
    plt.legend()
    plt.show()