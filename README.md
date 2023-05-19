# Requirements

Python 3.8+

```
pip install numpy
pip install Keras
pip install tensorflow
pip install Flask
pip install Werkzeug
pip install gunicorn
pip install pyttsx3
pip install matplotlib
pip install pandas
pip install tensorflow
pip install -U scikit-learn
pip install pyttsx3
```

# Installation

Download the code from the releases. Other files don't need to be downloaded manually.

With /code as the working directory, run any of the commands below.

Training results, partial datasets, and the model will be downloaded into the folder /training.

# How to run

To download the test and train data and train the modeL
```
  python main.py -t
  python main.py --train

  python main.py --train -m CNN -e 1
```

To show the results of the model:
```
  python main.py -r
  python main.py --results

  python code/main.py -r -m CNN
```
If a model isn't specified, it will show CNN by default.

Supported models:
* CNN (default)
* AlexNet
* MobileNet

To display help:
```
  python main.py -h
  python main.py --help
```

To download the raw data, split it into test and train datasets, and save them to file:
```
  python main.py -s
  python main.py --split
```

# File structure

```
|- data/
| |-- test/
| | |--  ...
| |-- train/
| | |--  ...
| |-- images.zip
|
|- program/
  |-- code/
  | |-- CustomArgParser.py
  | |-- Globals.py
  | |-- main.py
  | |-- Models.py
  |-- training/
    |-- CNN/
    | |-- traffic.h5
    | |-- traffic.pickle
    | |-- ModelAccuracy.png
    | |-- ModelLoss.png
    |-- AlexNet/
    | |-- traffic.h5
    | |-- traffic.pickle
    | |-- ModelAccuracy.png
    | |-- ModelLoss.png
    |-- x_test.npy
    |-- x_train.npy
    |-- y_test.npy
    |-- y_train.npy
```

# Contributions

* Shashidhar (@shashidhar788): Dataset, VGG19
* Jeet (@jeetparekh16): GUI, CNN, MobileNet
* Lakshmi (@Lakshmisatvika26): AlexNet, pyttsx3 text-to-speech integration
* Sally (@sallywei2): CLI/Program wrapper, documentation

Dataset: The German Traffic Sign Recognition Benchmark (GSTRB) by Institut Für Neuroinformatik (INI) https://benchmark.ini.rub.de/gtsrb_news.html