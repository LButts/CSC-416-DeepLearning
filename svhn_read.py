from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import scipy.io

mat = scipy.io.loadmat('train_32x32.mat')

#print(list(mat.keys()))

#print(mat['y'])
print(mat['X'])
