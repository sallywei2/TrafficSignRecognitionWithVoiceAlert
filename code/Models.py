"""
Module for keras CNN model.

Classes:
  CNN: custom keras CNN model, with custom functions for training, testing, 
       saving to file, and showing model results.
"""

import Globals as g
import pandas as pd
import numpy as np 
from PIL import Image
import pickle

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras # MobileNet
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing import image as kimg
from keras.utils.vis_utils import plot_model

import os
import pyttsx3

class Model:
  """
  Attributes:
    input_shape: keras Conv2D input_shape parameter
    model: internal saved keras model
    history: internal saved training history (keras History object)
    engine: pyttsx3 engine for text-to-speech

  Typical usage example:
    # create a new CNN for an input with keras input_shape of 10000
    model = CNN()
    model.new(input_shape=10000)
    
    # load from file
    model = CNN()
    model.load_model_from_file(file)

  Methods:
    load_model_from_file(filepath): Loads a saved CNN from file.
    load_history_from_file(pickle_location): Loads a saved pickle file containing the history of the saved CNN file.
    show_metrics(): Displays the metrics of the passed model
    test_model_on_image(imgpath): Test the model's classification on a single image
  """

  def __init__(self, model_type):
    self.valid_models = ['CNN','AlexNet']

    self.model = Sequential() # initialize basic placeholder model
    self.engine = pyttsx3.init() # text to speech engine

    self.history = -1
    self.type = 'CNN' # default type
    self.set_model_type(model_type)

  def AlexNet(self, input_shape):
    """
    Initializes the model as an AlexNet.
    # input shape: (30,30,3)
    """
    self.type = 'AlexNet'
    self.model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(30,30,3)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
        keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(44, activation='softmax')
    ])
    #Compilation of the model
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

  def CNN(self, input_shape):
    """
      Initializes the CNN with the following architecture and compiles it:
        Conv2D - 5x5 x 32 filters, ReLU
        Conv2D - 5x5 x 32 filters, ReLU
        MaxPool2D
        Dropout (25%)
        Conv2D - 3x3 x 64 filters, ReLU
        Conv2D - 3x3 x 64 filters, ReLU
        MaxPool2D - 2x2
        Dropout - 25%
        Flatten
        FC - 256, ReLU
        Dropout - 50%
        FC - 44, softmax
    """
    self.type='CNN'
    self.model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    self.model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    self.model.add(MaxPool2D(pool_size=(2, 2)))
    self.model.add(Dropout(rate=0.25))
    self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    self.model.add(MaxPool2D(pool_size=(2, 2)))
    self.model.add(Dropout(rate=0.25))
    self.model.add(Flatten())
    self.model.add(Dense(256, activation='relu'))
    self.model.add(Dropout(rate=0.5))
    self.model.add(Dense(44, activation='softmax'))
    # Compilation of the model
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
  def augment_data(self):
    """
    Performs image processing to augment the visual data for training MobileNet.
    Uses ImageDataGenerator. See also: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
    """
    #Augment the data
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.3, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True) # randomnly flip images

    # load and iterate training dataset
    train_id = datagen.flow_from_directory(g.DATASET + 'train', 
                                       target_size=(224,224), 
                                       color_mode='rgb', 
                                       class_mode="categorical")
    # load and iterate validation dataset
    valid_id = datagen.flow_from_directory(g.DATASET + 'test', 
                                      target_size=(224,224), 
                                      color_mode='rgb', 
                                      class_mode="categorical")
    return train_id, valid_id

  def train(self, X_train, y_train, batch_size, epochs, X_test, y_test):
    """
    Trains the model on the given dataset and returns the training history.
    """
    if self.type=='MobileNet':
      train_generator,val_generator = self.augment_data()
      self.history = self.model.fit(train_generator, validation_data=val_generator, batch_size=batch_size, epochs=epochs)
    else: # AlexNet or CNN
      self.history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
    return self.history

  def predict(self, img):
    """
    Predict probabilities for the input image
    """
    return self.model.predict(img)

  def save_model_to_file(self):
    """
    Save the keras model to file at the given filepath.
    """
    if not os.path.exists(self.get_root_fp()):
      os.mkdir(self.get_root_fp())
    filepath = self.get_model_fp()
    self.model.save(filepath)
    self.model.save_weights(self.get_root_fp() + g.WEIGHTS_FN)
    print("Model saved to file: %s" % filepath)

  def save_history_to_file(self):
    """
    Saves the training history to file as a pickle file.
    pickle_location: an absolute path
    """
    if not os.path.exists(self.get_root_fp()):
      os.mkdir(self.get_root_fp())
    pickle_location = os.path.abspath(self.get_history_fp())
    with open(pickle_location, 'wb') as file_pi:
      pickle.dump(self.history, file_pi, protocol=pickle.HIGHEST_PROTOCOL)
      print("History saved to file: %s" % pickle_location)

  def load_model_from_file(self):
    """
    Loads the model from file.
    """
    filepath = self.get_model_fp()
    print("Loading model from file: %s" % filepath)
    self.model = load_model(filepath) # keras.model.load_model
    #print("Model loaded from file.")
    return self.model

  def load_history_from_file(self):
    """
    Loads the history from file. Expects a pickle file.
    """
    pickle_location = self.get_history_fp()
    print("Loading training history from file: %s" % pickle_location)
    try:
      with open(pickle_location, "rb") as filepath:
        self.history = pickle.load(filepath)
        #print("Training history loaded from file.")
      return self.history
    except:
      # EOFError: Ran out of input (0KB file)
      print("  Warning: There was an issue when loading the training history file.")
      return

  def set_model_type(self, model_type):
    if model_type in self.valid_models:
      self.type = model_type
      print("Changed model type to %s" % self.type)
      return True
    return False

  def get_root_fp(self):
    """
    ROOT/CNN/
    """
    return g.ROOT + self.type + '\\'

  def get_root_url(self):
    """
    URL/ROOT_
    """
    return g.URL + self.type + '_'

  def get_model_fp(self):
    """
    ROOT/CNN/traffic.h5
    """
    return self.get_root_fp() + g.MODEL_FN
  
  def get_history_fp(self):
    """
    ROOT/CNN/traffic.pickle
    """
    return self.get_root_fp() + g.HISTORY_FN

  def get_model_url(self):
    """
    URL/CNN_traffic.h5
    """
    return self.get_root_url() + 'traffic.h5'

  def get_history_url(self):
    """
    URL/CNN_traffic.pickle
    """
    return self.get_root_url() + 'traffic.pickle'

  def show_img(self, filename):
    """
    Uses matplotlib to show an image (.png, etc.)
    """
    print("show file: ", filename)
    img = plt.imread(filename)
    imgplot = plt.imshow(img)
    plt.show()

  def show_metrics(self):
    """
      Shows and save to file the plots of training vs validation accuracy and loss for the model.
    """
    if self.history != -1:
      # accuracy 
      plt.figure(0)
      plt.plot(self.history.history['accuracy'], label='training accuracy')
      plt.plot(self.history.history['val_accuracy'], label='val accuracy')
      plt.title('Accuracy')
      plt.xlabel('epochs')
      plt.ylabel('accuracy')
      plt.legend()
      plt.savefig(self.get_root_fp() + g.MODELACC)
      plt.show()

      # Loss
      plt.plot(self.history.history['loss'], label='training loss')
      plt.plot(self.history.history['val_loss'], label='val loss')
      plt.title('Loss')
      plt.xlabel('epochs')
      plt.ylabel('loss')
      plt.legend()
      plt.savefig(self.get_root_fp()  + g.MODELLOSS)
      plt.show()
    else:
      # no saved history file
      self.show_img(self.get_root_fp() + g.MODELACC)
      self.show_img(self.get_root_fp() + g.MODELLOSS)
      
  def test_on_img(self, img):
    """
    Test the model on the given image (img)
    """
    data = []
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = self.model.predict_classes(X_test)
    return image,Y_pred

  def test_model_on_image(self, imgpath):
    """
    Test the model on a single image, given by its filepath
    """
    print("Predicting for a single image %s" % imgpath)

    #image = Image.open(imgpath)

    # Load and preprocess the image
    img = kimg.load_img(imgpath, target_size=(30, 30))
    img = kimg.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # Predict the class probabilities
    probabilities = self.model.predict(img)

    # Get the index with the highest probability
    predicted_class = np.argmax(probabilities)

    print("Predicted traffic sign is: ", g.CLASSES[predicted_class])
    self.engine.say(g.CLASSES[predicted_class])
    self.engine.runAndWait()

    plt.imshow(kimg.load_img(imgpath))
    plt.show()

    try:
      plot,prediction = self.test_on_img(imgpath)
      s = [str(i) for i in prediction] 
      a = int("".join(s)) 
      print("Predicted traffic sign is: ", g.CLASSES[a])
      plt.imshow(plot)
      plt.show()
    except:
      # AttributeError: 'Sequential' object has no attribute 'predict_classes'
      return