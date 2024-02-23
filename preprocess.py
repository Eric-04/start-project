import numpy as np
 
def preprocess_data(inputs: np.ndarray, labels: np.ndarray, split_percentage: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits data into training and testing data

    features: input features to be fed to a model. dimensions [num_samples, input_size]
    labels: labels associated with the input features. dimensions [num_samples,]
    split percentage: a decimal value representing the percentage of the number of samples which should become training samples
    returns: training features, testing features, training labels, testing labels
    """

    assert len(inputs) == len(labels)

    sample_size = len(inputs)

    if (split_percentage < 0 or split_percentage > 1): print("error: inccorrect percentage")
    split_size = int(sample_size * split_percentage)

    return inputs[0:split_size], inputs[split_size:], labels[0: split_size], labels[split_size:]