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
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing import image as kimg

class CNN:
  """
  # self.input_shape
  # self.model
  # self.history

  Usage:
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

  def __init__(self):
    self.model = Sequential() # initialize basic placeholder model
    
  def new(self, input_shape):
    """
      Creates a new CNN model and compiles it
    """
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

  def train(self, X_train, y_train, batch_size, epochs, X_test, y_test):
    # Trains the model on the given dataset and saves it to file
    self.history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
    
    model.save('..\\training\\traffics.h5')

    # TODO: save training information
    #pd.DataFrame(self.history).to_csv('..\\training\\traffics_history.csv')
    with open('/traffics_history', 'wb') as file_pi:
      pickle.dump(self.history.history, file_pi)

    return self.history

  def predict(self, img):
    # Predict probabilities for input image
    return self.model.predict(img)

  def load_model_from_file(self,filepath):
    self.model = load_model(filepath) # keras.model.load_model
    return self.model

  def load_history_from_file(self,pickle_location):
    try:
      with open(pickle_location, "rb") as filepath:
        self.history = pickle.load(filepath)
      return self.history
    except:
      return

  def show_metrics(self):
    """
      Shows plots of training vs validation accuracy and loss for the model.
    """
    if self.model.history:
      # accuracy 
      plt.figure(0)
      plt.plot(self.model.history.history['accuracy'], label='training accuracy')
      plt.plot(self.model.history.history['val_accuracy'], label='val accuracy')
      plt.title('Accuracy')
      plt.xlabel('epochs')
      plt.ylabel('accuracy')
      plt.legend()
      plt.show()

      # Loss
      plt.plot(self.history.history['loss'], label='training loss')
      plt.plot(self.history.history['val_loss'], label='val loss')
      plt.title('Loss')
      plt.xlabel('epochs')
      plt.ylabel('loss')
      plt.legend()
      plt.show()

  def test_on_img(self, img):
    # Test the model on the given image (img)
    data = []
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))
    X_test=np.array(data)
    Y_pred = self.model.predict_classes(X_test)
    return image,Y_pred

  def test_model_on_image(self, imgpath):
    # Test the model on a single image, given by its filepath
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