from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import scipy.io
import os
import sys
from six.moves.urllib.request import urlretrieve

DATA_PATH = "Data/"
PIXEL_DEPTH = 255
NUM_LABELS = 10
NUM_CHANELS = 3
last_percent_reported = None

#loads in a matlab file and separates the picture arrays from the labels
#then converts all of them to numpy arrays
def process_data(file):
    data = scipy.io.loadmat(file)
    pics = data['X']
    labels = data['y'].flatten().astype(np.int32)
    labels[labels==10] = 0
    #one_hot_labels = make_one_hot(labels)
    pic_array = make_pics_arr(pics)
    return pic_array, labels

#converts label array into array of one-hot labels
def make_one_hot(labels):
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return labels

#steps through the array and converts the individual channel arrays
#into 1 3d numpy array for each image
def make_pics_arr(pic_arr):
    rows = pic_arr.shape[0]
    cols = pic_arr.shape[1]
    chas = pic_arr.shape[2]
    num_pics = pic_arr.shape[3]
    scalar = 1/PIXEL_DEPTH

    new_arr = np.empty(shape=(num_pics, rows, cols, chas), dtype=np.float32)
    for x in range(0, num_pics):
        chas = pic_arr[:,:,:,x]
        norm_vec = (255-chas)*(1.0/255.0)
        norm_vec -= np.mean(norm_vec, axis=0)
        new_arr[x] = norm_vec

    return new_arr

#conditional statement used to ensure that the correct file is being loaded
def get_file_name(dataset):
    if dataset == "train":
        file_name = "train_32x32.mat"
    elif dataset == "test":
        file_name = "test_32x32.mat"
    elif dataset == "extra":
        file_name = "extra_32x32.mat"
    else:
        raise Exception('dataset needs to be either train, test, or extra')

    return file_name

#checks for necessary training and test files
#if the files aren't present it downloads them
#and then returns the data to be passed to the 
#processing function
def create_sets(dataset):
    df_name = get_file_name(dataset)
    df_pointer = os.path.join(DATA_PATH, df_name)

    if(not os.path.exists(df_pointer)):
        print("No file, checking for folder")
        if (not os.path.exists(DATA_PATH)):
            print("No directory, making one")
            os.makedirs(DATA_PATH)

    if os.path.isfile(df_pointer):
        print('File found, reading data')
        pull_data = readf(df_pointer)
        return pull_data

    else:
        new_file = do_down(DATA_PATH, df_name)
        return readf(new_file)

#opens the file, calls the proccessing function and then closes the file
def readf(file_name):
    file = open(file_name, 'rb')
    data = process_data(file)
    file.close()
    return data

#Downloads the desired files from the internet and 
#checks that the file was downloaded correctly into the
#correct place
def do_down(path, filename):
    base_url = "http://ufldl.stanford.edu/housenumbers/"
    print("Attempting download of:  ", filename)
    down_file, _ = urlretrieve(
        base_url + filename,
        os.path.join(path, filename),
        reporthook = prog_down)
    print("\nDownload Completed\n")
    statinfo = os.stat(down_file)
    if statinfo.st_size == get_expected_bytes(filename):
        print(down_file, "  found and verified")
    else:
        raise Exception("Failed to verify "+filename)
    return down_file

#outputs the progress of the download
def prog_down(count, block_size, total_size):
    global last_percent_reported
    percent = int(count * block_size * 100 / total_size)
    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()
        last_percent_reported = percent

#conditional holds the hardcoded value for each vile size to
#verify against after download
def get_expected_bytes(filename):
    if filename == "train_32x32.mat":
        byte_size = 182040794
    elif filename == "test_32x32.mat":
        byte_size = 64275384
    elif filename == "extra_32x32.mat":
        byte_size = 1329278602
    else:
        raise Exception("Invalid file name " + filename)
    return byte_size

#calls the set creator and file processors for both the training and test sets
def file_gen():
    train_data, train_labels = create_sets('train')
    write_npy_file(train_data, train_labels, 'train')

    test_data, test_labels = create_sets('test')
    write_npy_file(test_data, test_labels, 'test')

#calls the set creator and file processor for the extra set
#this is placed into a separate function because the extra set
#takes a long time for both downloading and processing
def extra_set_gen():
    extra_data, extra_labels = create_sets('extra')
    write_npy_file(extra_data, extra_labels, 'extra')

#loads the desired .npy file for both the pics and the labels then returns those arrays	
def load_data(setname):
    pics = np.load(os.path.join(DATA_PATH, setname+'_svhn_pics.npy'))
    labels = np.load(os.path.join(DATA_PATH, setname+'_svhn_labels.npy'))
    return pics, labels

#takes the processed arrays and writes them to .npy files, fails if the file already exists	
def write_npy_file(data_array, label_array, setname):
    ndat_pointer = os.path.join(DATA_PATH, setname+'_svhn_pics.npy')
    nlabel_pointer = os.path.join(DATA_PATH, setname+'_svhn_labels.npy')

    if(not os.path.exists(ndat_pointer)):
        np.save(ndat_pointer, data_array)
        print('Saving to %s_svhn_pics.npy file done.' %setname)
    else:
        print("File already converted to numpy array")

    if(not os.path.exists(nlabel_pointer)):
        np.save(nlabel_pointer, label_array)
        print('Saving to %s_svhn_labels.npy file done.\n' %setname)
    else:
        print("File already converted to numpy array\n")

if __name__ == "__main__":
    file_gen()
