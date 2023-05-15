global ROOT
global DATA
global TARGET
global SAVED_MODEL
global SAVED_HISTORY
global TEST_IMAGE

global CLASSES

DATASET = '..\\..\\data\\'
IMG = 'train\\14\\00014_00006_00029.png'
TEST_IMAGE = DATASET + IMG
IMAGES = DATASET + 'images.zip' 

ROOT = '..\\training\\'
# TODO: static training and test data, rather than full data & random split
DATA = ROOT + 'data.npy'
TARGET = ROOT + 'target.npy'
MODEL_FP = ROOT + 'traffic.h5' # saved keras model 
HISTORY_FP = ROOT + 'traffic.pickle' # location of pickle

X_TRAIN = ROOT + 'x_train.npy'
Y_TRAIN = ROOT + 'y_train.npy'
X_TEST = ROOT + 'x_test.npy'
Y_TEST = ROOT + 'y_test.npy'

URL = 'https://github.com/sallywei2/Deep-Learning-With-Traffic-Signs-Under-Different-Weather-Conditions/releases/download/v0.0.0-alpha/'
X_TRAIN_URL = URL + 'x_train.npy'
Y_TRAIN_URL = URL + 'y_train.npy'
X_TEST_URL = URL + 'x_test.npy'
Y_TEST_URL = URL + 'y_test.npy'
MODEL_URL = URL + 'traffic.h5'
HISTORY_URL = URL + 'traffic.pickle'
IMAGES_URL = URL + 'images.zip'

CLASSES = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons',
            43: 'No sign'}