import numpy as np
import h5py

# Function used for loading the data
def load_dataset():

    # Dataset is the catvnoncat dataset from Kaggle
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # Train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # Train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # Test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # Test set labels
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig
