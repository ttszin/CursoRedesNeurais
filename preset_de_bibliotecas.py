# -*- coding: utf-8 -*-
"""
Created on Thu May  2 08:25:01 2024

@author: Matheus
"""

!pip install -q -U tensorflow
!pip install -q -U keras
!pip install -q -U numpy
!pip install -q -U pandas
!pip install -q -U tensorflow-addons
!pip install -q -U keras-utils


import os
import keras
import numpy as np
import cv2
import PIL
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Convolution2D as conv
from math import sqrt
from PIL import Image
from tensorflow import keras
from numpy import mean
from keras.models import Sequential, load_model
from tensorflow.keras import regularizers, layers, Model
from keras.preprocessing import image
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
from keras import backend as B
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras import Model
from keras.layers import Dense, Add, Conv1D, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout, ZeroPadding2D, MaxPooling2D, Activation, Input, UpSampling2D, AveragePooling2D, Reshape, InputLayer, SeparableConv2D
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from keras.regularizers import l2
from keras.initializers import glorot_uniform
from skimage.feature import peak_local_max
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.exposure import equalize_adapthist
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel, rank, median
from skimage.util import img_as_ubyte
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, accuracy_score
from tensorflow.keras.metrics import *
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support, f1_score
