"""Custom argument parser"""

import argparse

class CustomArgParser():
    """
    This class defines the parser for this project.

    https://docs.python.org/3/library/argparse.html
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(
                    prog='main',
                    description='This is some text describing what this program does.',
                    epilog='Mapillary Traffic Signs: https//www.mapillary.com/dataset/trafficsign')

        # specify arguments
        self.parser.add_argument('-t','--train'
                    , action='store_true' # set to True if this argument is present
                    , help='download training data set and train the model')
        self.parser.add_argument('-r','--results'
                    , action='store_true'
                    , help='show results after validation and testing')

    def parse_args(self):
        return self.parser.parse_args()

