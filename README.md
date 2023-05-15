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
```

To show the results of the model:
```
  python main.py -r
  python main.py --results
```

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


# Contributors

* CNN model training & tuning: @jeetparekh16
* Program wrapper: @sallywei2
* WIP

Dataset: The German Traffic Sign Recognition Benchmark (GSTRB) by Institut Für Neuroinformatik (INI) https://benchmark.ini.rub.de/gtsrb_news.html