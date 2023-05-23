"""
Module containing the main program that handles passed arguments.
"""
from CustomArgParser import CustomArgParser
from Models import Model
import Globals as g

import numpy as np
from sklearn.model_selection import train_test_split
import cv2

from keras.utils import to_categorical

import os, sys
import requests
import zipfile
from tqdm import tqdm

def download_from_internet(file_url, fp, force_download=False):
    """
    Downloads the file at file_url and saves it to file at fp.
    Only downloads if the specified destination (fp) does not exist.
    Returns quietly otherwise.
    Source: https://www.geeksforgeeks.org/downloading-files-web-using-python/
    """
    if (not os.path.exists(fp)) or (force_download):
        print("Downloading %s" % fp)
        r = requests.get(file_url, stream = True)
        if r.status_code == 200:
            with open(fp,"wb") as file:
                for chunk in r.iter_content(chunk_size=1024):
                     if chunk:
                         file.write(chunk) # write to file
        else:
            g.print_debug("Could not download the requested file: %s" % file_url)

def download_images(force_download=False):
    """
    Downloads the raw images from the internet and unzips them if the folder doesn't exist.
    Unzip files: https://stackoverflow.com/a/3451150
    """
    download_from_internet(g.IMAGES_URL, g.IMAGES)
    if not os.path.exists(g.DATASET + "\\train"):
        try:
            print("Extracting %s" % g.IMAGES)
            with zipfile.ZipFile(g.IMAGES, 'r') as zip_ref:
                zip_ref.extractall(g.DATASET)
        except:
            if force_download:
                print("Redownload failed. Please check the images zipfile manually.")
                return
            else:
                print("There was an issue while extracting the zip file. Attempting to redownload....")
                os.remove(g.IMAGES)
                download_images(force_download=True)
                exit()

def confirm_preprocessed_images(_df, _rowcnt, _srcdir):
    '''
    Crop 처리가 잘 되는지 눈으로 확인
    Integrated from notebooks/german-traffic-signs-preprocessing.ipynb
    '''
    _df_size = len(_df.index)
    if (_df_size < _rowcnt):
        _rowcnt = _df_size
        
    for _, _row in _df.sample(_rowcnt).iterrows():
        _filename = _row['Path'].lower()
        _classId = _row['ClassId']
        _x1 = _row['Roi.X1']
        _x2 = _row['Roi.X2']
        _y1 = _row['Roi.Y1']
        _y2 = _row['Roi.Y2']
    
        _img = cv2.imread(_srcdir + _filename)
        _crop_img = _img[_y1:_y2, _x1:_x2]
    
        _f, _ax = plt.subplots(1, 2, figsize=(5,12))
        _ax[0].imshow(_img)
        _ax[0].set_title('original')
        _ax[1].imshow(_crop_img)
        _ax[1].set_title('cropped')
        plt.show()

def resize_images():
    """
    Resize images into 32x32(x3) for VGG
    """
    for i in tqdm(range(len(g.CLASSES))):
        path = os.path.join(cur_path, 'train', str(i))
        images = os.listdir(path)
        for a in images:
            try:
                image = Image.open(path + '/' + a)
                image = image.resize((32, 32))
                image = np.array(image)
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(e)
    return data, labels

def write_preprocessed_images(_df, _srcdir, _outdir):
    '''
    Crop 처리된 이미지를 저장한다.
    Integrated from notebooks/german-traffic-signs-preprocessing.ipynb
    '''
    for _, _row in _df.iterrows():
        _filename = _row['Path'].lower()
        _classId = _row['ClassId']
        _x1 = _row['Roi.X1']
        _x2 = _row['Roi.X2']
        _y1 = _row['Roi.Y1']
        _y2 = _row['Roi.Y2']
    
        _img = cv2.imread(_srcdir + _filename)
        _crop_img = _img[_y1:_y2, _x1:_x2]
    
        if not cv2.imwrite(_outdir + _filename, _crop_img):
            raise Exception("Could not write image: {}".format(_filename))

def preprocess_images():
    """
    This script pre-processes the German Traffic Sign Recognition Benchmark (GTSRB) available at Kaggle (https://www.kaggle.com/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).
    The original dataset contains image files as well as metadata files. As the images capture scenes larger than the traffic signs, the metadata files contain coordinates to 
    locate the traffic sign within each image. By cropping images with these coordinates, I expect I can reduce noise in data and enhance training result.
    
    Integrated from notebooks/german-traffic-signs-preprocessing.ipynb
    """
    SOURCE_DIR = g.DATASET #'/notebooks/gtsrb/'
    OUT_DIR = g.DATASET #'/notebooks/gtsrb-preprocessed/'

    download_from_internet(g.IMG_META_TRAIN_URL,g.IMG_META_TRAIN_FP)
    TRAIN_META_FILE = g.IMG_META_TRAIN_FP

    download_from_internet(g.IMG_META_TEST_URL,g.IMG_META_TEST_FP)
    TEST_META_FILE = g.IMG_META_TEST_FP

    df_train = pd.read_csv(TRAIN_META_FILE, delimiter=',')
    #print(df_train.shape)
    #df_train.head()

    df_test = pd.read_csv(TEST_META_FILE, delimiter=',')
    #print(df_test.shape)
    #df_test.head()

    write_preprocessed_images(df_train, _srcdir=SOURCE_DIR, _outdir=OUT_DIR)
    write_preprocessed_images(df_test, _srcdir=SOURCE_DIR, _outdir=OUT_DIR)
    return

def split_raw_data(download_data=False):
    """
    Splits the saved full dataset into testing and training data and saves it to file.
    """
    if download_data:
        data=np.load(g.DATA)
        labels=np.load(g.TARGET)
    else:
        data, labels = resize_images()

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
    if model_type == 'AlexNet':
        model.AlexNet(input_shape=input_size)
    if model_type == 'VGG':
        model.VGG(input_shape=input_size)
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

    if not os.path.exists(model.get_root_fp()):
        os.mkdir(model.get_root_fp())

    download_from_internet(model.get_model_url(), model.get_model_fp())
    download_from_internet(model.get_history_url(), model.get_history_fp())
        
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
        model_type.replace("'","").replace(":","") #clean input
        print("Selected model: %s" % model_type)

    if vargs['epochs']:
        epochs = int(vargs['epochs'])
        print("Training for custom number of epochs: %d" % epochs)
    else:
        epochs = 15

    if vargs['train'] == True:
        download_images()
        preprocess_images()
        X_train, X_test, y_train, y_test = download_data()
        train_model(X_train, X_test, y_train, y_test, epochs, model_type)
    elif vargs['results'] == True:
        model = load_saved_model(model_type)
        if model.history == -1:
            download_from_internet(model.get_root_url() + g.MODELACC, model.get_root_fp() + g.MODELACC)
            download_from_internet(model.get_root_url() + g.MODELLOSS, model.get_root_fp() + g.MODELLOSS)
        model.show_metrics() # plot metrics
        
        download_images()
        model.test_model_on_image(g.TEST_IMAGE) # test on a single image
    elif vargs['split'] == True:
        download_images()
        split_raw_data()

if __name__ == '__main__':
    main()