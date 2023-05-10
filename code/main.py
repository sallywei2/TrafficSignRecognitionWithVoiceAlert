from CustomArgParser import CustomArgParser
import Models as Model
import Globals as g

import numpy as np
from tensorflow.keras.preprocessing import image

import os

def download_data():
    #TODO: Download static training and testing data
    data=np.load(g.ROOT + g.DATA)
    labels=np.load(g.ROOT + g.TARGET)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, X_test, y_train, y_test):
    # Train and save the model
    input_size = X_train.shape[1:]
    model = CNN(input_size=input_size)
    model.train(X_train, y_train, 32, epochs, X_test, y_test)

def load_saved_model():
    print("Loading saved model from file...")
    model = Model.load_model_from_file(g.ROOT + g.SAVED_MODEL)
    history = Model.load_history_from_file(g.ROOT + g.SAVED_HISTORY)
    model.history = history
    print("Model loaded from file.")
    return model

def show_model_results(model):
    # Show model results
    Model.show_metrics(model)

def main(*args):

    parser = CustomArgParser()
    args = parser.parse_args()
    vargs = vars(args)

    if vargs['train'] == True:
        X_train, X_test, y_train, y_test = download_data()
        train_model(X_train, X_test, y_train, y_test)
    elif vargs['results'] == True:
        model = load_saved_model()
        show_model_results(model) # plot metrics
        # test model
        Model.test_model_on_image(model, g.TEST_IMAGE)

if __name__ == '__main__':
    main()