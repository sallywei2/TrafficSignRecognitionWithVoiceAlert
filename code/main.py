"""
Module containing the main program that handles passed arguments.
"""
from CustomArgParser import CustomArgParser
from Models import Model
import Globals as g

import numpy as np
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

import os, sys
import requests
import zipfile

def download_from_internet(file_url, fp):
    """
    Downloads the file at file_url and saves it to file at fp.
    Only downloads if the specified destination (fp) does not exist.
    Returns quietly otherwise.
    Source: https://www.geeksforgeeks.org/downloading-files-web-using-python/
    """
    if not os.path.exists(fp):
        print("Downloading %s" % fp)
        r = requests.get(file_url, stream = True)
        with open(fp,"wb") as file:
            for chunk in r.iter_content(chunk_size=1024):
                 if chunk:
                     file.write(chunk) # write to file

def download_images():
    """
    Downloads the raw images from the internet and unzips them if the folder doesn't exist.
    Unzip files: https://stackoverflow.com/a/3451150
    """
    download_from_internet(g.IMAGES_URL, g.IMAGES)
    if not os.path.exists(g.DATASET + "\\train"):
        print("Extracting %s" % g.IMAGES)
        with zipfile.ZipFile(g.IMAGES, 'r') as zip_ref:
            zip_ref.extractall(g.DATASET)

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

    download_from_internet(g.X_TRAIN_URL, g.X_TRAIN)
    download_from_internet(g.X_TEST_URL, g.X_TEST)
    download_from_internet(g.Y_TRAIN_URL, g.Y_TRAIN)
    download_from_internet(g.Y_TEST_URL, g.Y_TEST)

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

def train_model(X_train, X_test, y_train, y_test, epochs, model_type):
    """
    Train and save the model's training history to file
    """
    input_size = X_train.shape[1:]
    model = Model(model_type)
    if model_type =='AlexNet':
        model.AlexNet(input_shape=input_size)
    else:
        model.CNN(input_shape=input_size)
    history = model.train(X_train, y_train, 32, epochs, X_test, y_test)

    model.save_model_to_file()
    model.save_history_to_file()

def load_saved_model(model_type='CNN'):
    """
    Loads the saved model and training history from file, or from the internet if it's not available.
    If no model is specified, it loads the CNN by default.
    See Models.Model for valid types.
    """
    model = Model(model_type)
    download_from_internet(model.get_model_url(), model.get_model_fp())
    download_from_internet(model.get_model_fp(), model.get_history_fp())
        
    model.load_model_from_file()
    model.load_history_from_file()
    return model

def main(*args):

    parser = CustomArgParser()
    args = parser.parse_args()
    vargs = vars(args)

    os.chdir(os.path.dirname(os.path.realpath(sys.argv[0])))
    if not os.path.exists(g.ROOT):
      os.mkdir(g.ROOT)
    if not os.path.exists(g.DATASET):
      os.mkdir(g.DATASET)

    if vargs['model']:
        model_type = vargs['model']
        print("Selected model: %s" % model_type)

    if vargs['epochs']:
        epochs = int(vargs['epochs'])
        print("Training for custom number of epochs: %d" % epochs)
    else:
        epochs = 15

    if vargs['train'] == True:
        download_images()
        X_train, X_test, y_train, y_test = download_data()
        train_model(X_train, X_test, y_train, y_test, epochs, model_type)
    elif vargs['results'] == True:
        download_images()
        model = load_saved_model(model_type)
        model.show_metrics() # plot metrics
        model.test_model_on_image(g.TEST_IMAGE) # test on a single image
    elif vargs['split'] == True:
        split_raw_data()

if __name__ == '__main__':
    main()