"""
Module containing the main program that handles passed arguments.
"""
from CustomArgParser import CustomArgParser
import Models as Model
import Globals as g

import numpy as np
from sklearn.model_selection import train_test_split

import os

def download_data():
    """
    TODO: Download static training and testing data
    """
    data=np.load(g.ROOT + g.DATA)
    labels=np.load(g.ROOT + g.TARGET)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    """
    # Train and save the model
    """
    input_size = X_train.shape[1:]
    model = Model.CNN()
    model.new(input_size=input_size)
    model.train(X_train, y_train, 32, epochs, X_test, y_test)

def load_saved_model():
    """
    Loads the saved model and training history from file.
    """
    print("Loading saved model from file...")
    model = Model.CNN()
    model.load_model_from_file(g.ROOT + g.SAVED_MODEL)
    model.load_history_from_file(g.ROOT + g.SAVED_HISTORY)
    print("Model loaded from file.")
    return model

def main(*args):

    parser = CustomArgParser()
    args = parser.parse_args()
    vargs = vars(args)

    if vargs['train'] == True:
        X_train, X_test, y_train, y_test = download_data()
        train_model(X_train, X_test, y_train, y_test)
    elif vargs['results'] == True:
        model = load_saved_model()
        model.show_metrics() # plot metrics
        model.test_model_on_image(g.TEST_IMAGE) # test on a single image

if __name__ == '__main__':
    main()