"""
Module containing the main program that handles passed arguments.
"""
from CustomArgParser import CustomArgParser
import Models as Model
import Globals as g

import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

import os, sys

    """
    """
def split_raw_data():
    """
    Splits the saved full dataset into testing and training data and saves it to file.
    """
    data=np.load(g.DATA)
    labels=np.load(g.TARGET)

    print("Raw data loaded from %s" % g.DATA)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    
    # save the split data to file
    np.save(g.X_TRAIN, X_train)
    np.save(g.X_TEST, X_test)
    np.save(g.Y_TRAIN, y_train)
    np.save(g.Y_TEST, y_test)

    print("Received:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print("Expected: (62734, 30, 30, 3) (15684, 30, 30, 3) (62734,) (15684,)")

    return X_train, X_test, y_train, y_test

def load_splits_from_file():
    """
    Load the training and testing split data from file.
    """
    print("Training data loaded from %s" % g.ROOT)

    X_train=np.load(g.X_TRAIN)
    X_test=np.load(g.X_TEST)
    y_train=np.load(g.Y_TRAIN)
    y_test =np.load(g.Y_TEST)

    return X_train, X_test, y_train, y_test

def download_data():
    """
    Downloads the training and testing data.
    """
    if not os.path.exists(g.ROOT):
      os.mkdir(g.ROOT)

    X_train, X_test, y_train, y_test = load_splits_from_file()

    num_classes = len(g.CLASSES)
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test, epochs):
    """
    Train and save the model's training history to file
    """
    input_size = X_train.shape[1:]
    model = Model.CNN()
    model.new(input_shape=input_size)
    history = model.train(X_train, y_train, 32, epochs, X_test, y_test)

    if not os.path.exists(g.ROOT):
      os.mkdir(g.ROOT)
    model.save_model_to_file(g.MODEL_FP)

    if not os.path.exists(g.HISTORY_FP):
      os.mkdir(g.HISTORY_FP)
    model.save_history_to_file(os.path.abspath(g.HISTORY_FP)) # expects an absolute path

def load_saved_model():
    """
    Loads the saved model and training history from file.
    """
    model = Model.CNN()
    model.load_model_from_file(g.MODEL_FP)
    model.load_history_from_file(g.HISTORY_FP)
    return model

def main(*args):

    parser = CustomArgParser()
    args = parser.parse_args()
    vargs = vars(args)

    os.chdir(os.path.dirname(os.path.realpath(sys.argv[0])))

    if vargs['train'] == True:
        epochs = 15
        X_train, X_test, y_train, y_test = download_data()
        train_model(X_train, X_test, y_train, y_test, epochs)
    elif vargs['results'] == True:
        model = load_saved_model()
        model.show_metrics() # plot metrics
        model.test_model_on_image(g.TEST_IMAGE) # test on a single image
    elif vargs['split'] == True:
        split_raw_data()

if __name__ == '__main__':
    main()