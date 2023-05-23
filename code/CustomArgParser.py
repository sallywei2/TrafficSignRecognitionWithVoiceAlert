"""
Custom argument parser

Classes:
    CustomArgParser: Custom ArgumentParser with arguments specific to this program
"""

import argparse

class CustomArgParser():
    """This class defines the parser for this project.

    Attributes:
        parser (argparser.ArgumentParser)

    Typical usage example:

        parser = CustomArgParser()
        args = parser.parse_args()

    See also https://docs.python.org/3/library/argparse.html
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
                    prog='main',
                    description='Traffic Sign Recognition.',
                    epilog='Dataset: The German Traffic Sign Recognition Benchmark https://benchmark.ini.rub.de/gtsrb_news.html')

        # specify arguments
        self.parser.add_argument('-t','--train'
                    , action='store_true' # set to True if this argument is present
                    , help='download training data set and train the model')
        self.parser.add_argument('-r','--results'
                    , action='store_true'
                    , help='show results after validation and testing')
        self.parser.add_argument('-s','--split'
                    , action='store_true' # set to True if this argument is present
                    , help='download the raw data, split it into training and test, and save to file')
        self.parser.add_argument('-m','--model'
                    , help='Specify which model to run. If unspecified, defaults to CNN. Valid options: CNN, AlexNet, VGG')
        self.parser.add_argument('-e','--epochs'
                    , help='Specify how many epochs to train for. The default is 15.')


    def parse_args(self):
        return self.parser.parse_args()

