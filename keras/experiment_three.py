import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import glob, os, sys, random

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from pti01dataset import PTI01Dataset

import getpass
username = getpass.getuser()
# DATASET_PATH = '/home/gustavo/workspace/datasets/pti/PTI01/'
# DATASET_PATH = '/home/grvaliati/workspace/datasets/pti/PTI01/'
DATASET_PATH = '/home/{}/workspace/datasets/pti/PTI01/'.format(username)

dataset = PTI01Dataset(dataset_path=DATASET_PATH)

print("Defining model...")
model = Sequential()
# input: images with 3 channels -> E.g. (480, 640, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=dataset.get_img_shape()))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)
model.summary()
print("Done.")

train_gen = dataset.create_train_gen()
model.fit_generator(train_gen, steps_per_epoch=dataset.get_steps_per_epoch(), epochs=2)

# from keras.callbacks import EarlyStopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# model.fit(x, y, validation_split=0.2, callbacks=[early_stopping])

# model.fit(x_train, y_train, batch_size=32, epochs=10)
