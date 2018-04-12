from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import scipy.io
import os
from six.moves.urllib.request import urlretrieve

DATA_PATH = "Data/"
PIXEL_DEPTH = 255
NUM_LABELS = 10
NUM_CHANELS = 3
blank = 0

def process_data(file):
    data = scipy.io.loadmat(file)
    pics = data['X']
    labels = data['y'].flatten()
    labels[labels==10] = 0
    one_hot_labels = make_one_hot(labels)
    pic_array = make_pics_arr(pics)
    return pic_array, one_hot_labels

def make_one_hot(labels):
    labels = (np.arrange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return labels

def make_pics_arr(pic_arr):
    rows = pics_arr.shape[0]
    cols = pics_arr.shape[1]
    chas = pics_arr.shape[2]
    num_pics = pics_arr.shape[3]
    scalar = 1/PIXEL_DEPTH

    new_arr = np.empty(shape=(num_pics, rows, cols, chas), stype=np.float32)
    for x in range(0, num_pics):
        chas = pic_arr[:,:,:,x]
        norm_vec = (255-chas)*(1.0/255.0)
        norm_vec -= np.mean(norm_vec, axis=0)
        new_arr[x] = norm_vec

    return new_arr

def get_file_name(dataset):
    if dataset == "train":
        file_name = "train_32x32.mat"
    elif dataset == "test":
        file_name = "test_32x32.mat"
    elif dataset == "extra":
        file_name = "extra_32x32.mat"
    else
        raise Exception('dataset needs to be either train, test, or extra')

    return file_name

def create_sets(dataset):
    df_name = get_file_name(DATA_PATH, dataset)
    df_pointer = os.path.join(path, df_name)

    if(not os.path.exists(df_pointer)):
        if (not os.path.exists(DATA_PATH)):
            os.makedirs(DATA_PATH)

    if os.path.isfile(df_pointer):
        pull_data = readf(df_pointer)
        return pull_data

    else:
        new_file = do_down(DATA_PATH, df_name)
        return readf(new_file)

def readf(file_name):
    file = open(file_name, 'rb')
    data = process_data(file)
    file.close()

def do_down(path, filename):
    base_url = "http://ufldl.stanford.edu/housenumbers/"
    print "Attempting download of:  ", filename
    down_file, _ = urlretrieve(
        base_url + filename,
        os.path.join(path, filename),
        reporthook = prog_down)
    print "\n Download Completed\n"
    statinfo = os.stat(down_file)
    if statinfo.st_size == get_expected_bytes(filename):
        print(down_file, "  found and verified")
    else:
        raise Exception("Failed to verify "+filename)
    return down_file

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

def file_gen():
    train_data, train_labels = create_sets('train')
    write_npy_file(train_data, train_labels, 'train')

    test_data, test_labels = create_sets('test')
    write_npy_file(test_data, test_labels, 'test')

def load_data(setname):
    pics = np.load(os.path.join(DATA_PATH, setname+'_svhn_pics.npy'))
    labels = np.load(os.path.join(DATA_PATH, setname+'_svhn_labels.npy'))
    return pics, labels

def write_npy_file(data_array, label_array, setname):
    np.save(os.path.join(DATA_PATH, setname+'_svhn_pics.npy'), data_array)
    print('Saving to %s_svhn_pics.npy file done.' %setname)
    np.save(os.path.join(DATA_PATH, setname+'_svhn_labels.npy'), label_array)
    print('Saving to %s_svhn_labels.npy done.' %setname)
