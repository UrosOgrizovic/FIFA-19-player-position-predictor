# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf
import tensorflow.keras as keras


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import model.ANN as ANN
import data.data as DATA
import plot.plot as PLOT

data = DATA.read_data()
# number of positions = 27
# ?GK column is excluded so as to avoid the dummy variable trap?
positions = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM",
             "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM",
             "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB", "GK"]

sections= ["ATT", "MID", "DEF", "GK"]

attacking_positions = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW"]
midfield_positions = ["LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LDM", "CDM", "RDM"]
defensive_positions = ["LWB", "RWB", "LB", "LCB", "CB", "RCB", "RB"]

players = DATA.make_players(data)

# PLOT.plot_number_of_players_by_position(players)

# randomly splitting data into train/test sets in 80/20 ratio
x_train, x_test = train_test_split(players, test_size=0.2)




# randomly splitting training data into train/validation sets in 80/20 ratio
x_train, x_val = train_test_split(x_train, test_size=0.2)

# 1. labels have to be numerical
# 2. one-hot encoding has to be used instead of integer encoding, so that the model doesn't assume a natural ordering between categories
positions_train, positions_val, positions_test = [], [], []
sections_train, sections_val, sections_test = [], [], []

# first, the categorical data has to be mapped to integer data
DATA.populate_positions(x_train, positions_train)
DATA.populate_sections(x_train, sections_train)

# now, the integer data can be one-hot encoded
positions_train = keras.utils.to_categorical(positions_train)
sections_train = keras.utils.to_categorical(sections_train)

x_train = x_train.drop(["Position"], axis=1)

# normalization is used instead of standardization, because the data is not Gaussian
x_train = keras.utils.normalize(x_train.values, axis=1)

###

DATA.populate_positions(x_val, positions_val)
DATA.populate_sections(x_val, sections_val)

positions_val = keras.utils.to_categorical(positions_val)
sections_val = keras.utils.to_categorical(sections_val)

x_val = x_val.drop(["Position"], axis=1)
x_val = keras.utils.normalize(x_val.values, axis=1)

###

DATA.populate_positions(x_test, positions_test)
DATA.populate_sections(x_test, sections_test)

positions_test = keras.utils.to_categorical(positions_test)
sections_test = keras.utils.to_categorical(sections_test)

x_test = x_test.drop(["Position"], axis=1)
x_test = keras.utils.normalize(x_test.values, axis=1)

###
model_positions = ANN.make_NN(x_train, 27)
dashes = "-"*50
print(dashes + " Positions model training " + dashes)
ANN.train_NN(model_positions, x_train, positions_train, num_of_epochs=10, batch_size=100)
print(dashes*2)

model_sections = ANN.make_NN(x_train, 4)
print(dashes + " Sections model training " + dashes)
ANN.train_NN(model_sections, x_train, sections_train, class_weights={0: 1., 1: 2, 2: 2, 3: 1.}, num_of_epochs=10, batch_size=100)
print(dashes*2)

print(dashes + " Positions model validation " + dashes)
val_loss_positions, val_acc_positions = model_positions.evaluate(x_val, positions_val)  # evaluate the out of sample data with model_positions
print(dashes*2)
print(dashes + " Sections model validation " + dashes)
val_loss_sections, val_acc_sections = model_sections.evaluate(x_val, sections_val)  # evaluate the out of sample data with model_sections
print(dashes*2)

predictions_positions = model_positions.predict(x_test)
predictions_sections = model_sections.predict(x_test)
# DATA.display_predictions(predictions_sections, sections_test, "SECTION")
#DATA.display_predictions(predictions_positions, positions_test, "POSITION")