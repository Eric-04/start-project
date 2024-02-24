import numpy as np
import tensorflow as tf
 
def preprocess_data(inputs, labels, split_percentage: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into training and testing data
    """

    assert len(inputs) == len(labels)

    sample_size = len(inputs)

    if (split_percentage < 0 or split_percentage > 1): print("error: percentage out of bounds")
    split_size = int(sample_size * split_percentage)

    # Convert labels to one-hot encoding
    label_to_index = {label: i for i, label in enumerate(set(labels))}
    num_classes = len(label_to_index)
    labels = np.array([label_to_index[label] for label in labels])
    labels = tf.one_hot(labels, num_classes)
    
    return inputs[:split_size], inputs[split_size:], labels[:split_size], labels[split_size:]