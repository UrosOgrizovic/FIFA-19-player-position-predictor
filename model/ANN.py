import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


import os
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def make_NN(x_train, num_of_output_neurons):
    model = keras.models.Sequential()

    # a simple fully-connected layer, 128 units, relu activation
    model.add(Dense(128, kernel_initializer='random_normal', input_dim=len(x_train[0])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(128, kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # output layer. 27 units for 27 classes (= positions) or 4 units for 4 classes (= sections).
    # Softmax for probability distribution
    model.add(Dense(num_of_output_neurons, kernel_initializer='random_normal'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

def train_NN(model, x_train, y_train, class_weights=None, num_of_epochs=None, batch_size=None):
    adam = keras.optimizers.Adam(lr=0.01)

    # Adam = RMSProp + Momentum
    # categorical_crossentropy should be used for one-hot encoded data
    model.compile(optimizer=adam,  # Good default optimizer to start with
                  loss='categorical_crossentropy',
                  # how will we calculate our "error." Neural network aims to minimize loss.
                  metrics=['accuracy'])  # what to track

    if class_weights is not None:
        model.fit(x_train, y_train, validation_split=0.2, epochs=num_of_epochs, batch_size=batch_size, class_weight=class_weights)  # train the model
    else:
        model.fit(x_train, y_train, validation_split=0.2, epochs=num_of_epochs, batch_size=batch_size)  # train the model

#TODO: try RandomForest (sklearn lib, ensemble package)

