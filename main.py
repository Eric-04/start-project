from data import generate_data
from preprocess import preprocess_data
import numpy as np

if __name__ == "__main__":
    """
    Read in Diabetes data, initialize your model, and train and test your model.
    """
    inputs, labels = generate_data()

    train_inputs, test_inputs, train_labels, test_labels = preprocess_data(np.array(inputs), np.array(labels), 0.8)

    # args = get_single_layer_model_components()

    # train_losses = args.model.fit(
    #     train_inputs,
    #     train_labels,
    #     epochs=args.epochs,
    # )

    # test_losses = args.model.evaluate(
    #     test_inputs,
    #     test_labels,
    # )