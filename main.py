# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

data = pd.read_csv('input/data.csv')
# number of positions = 27
# ?GK column is excluded so as to avoid the dummy variable trap?
positions = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM",
             "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM",
             "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB", "GK"]
# there's no difference between lw and rw, so positions can be grouped
grouped_positions = ["ST", "ST", "ST", "LW/RW", "CF", "CF", "CF", "LW/RW",
                     "AM", "AM", "AM", "LM/RM", "CM", "CM", "CM", "LM/RM",
                     "LB/RB", "DM", "DM", "DM", "LB/RB", "LB/RB",
                     "CB", "CB", "CB", "LB/RB", "GK"]

sections= ["ATT", "MID", "DEF", "GK"]

attacking_positions = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW"]
midfield_positions = ["LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LDM", "CDM", "RDM"]
defensive_positions = ["LWB", "RWB", "LB", "LCB", "CB", "RCB", "RB"]


players = data[["Position", "Skill Moves", "Crossing", "Finishing",
               "HeadingAccuracy", "ShortPassing", "Volleys",
               "Dribbling", "Curve", "FKAccuracy", "LongPassing", "BallControl",
               "Acceleration", "SprintSpeed", "Agility", "Reactions",
               "Balance", "ShotPower", "Jumping", "Stamina", "Strength",
               "LongShots", "Aggression", "Interceptions", "Positioning",
               "Vision", "Penalties", "Composure", "Marking",
               "StandingTackle", "SlidingTackle",
               "GKDiving", "GKHandling", "GKKicking", "GKPositioning",
               "GKReflexes"]]
#players.head()
#players.isnull().sum()

players["Skill Moves"].fillna(1, inplace = True)
players["Crossing"].fillna(1, inplace = True)
players["Finishing"].fillna(1, inplace = True)
players["HeadingAccuracy"].fillna(1, inplace = True)
players["ShortPassing"].fillna(1, inplace = True)
players["Volleys"].fillna(1, inplace = True)
players["Dribbling"].fillna(1, inplace = True)
players["Curve"].fillna(1, inplace = True)
players["FKAccuracy"].fillna(1, inplace = True)
players["LongPassing"].fillna(1, inplace = True)
players["BallControl"].fillna(1, inplace = True)
players["Acceleration"].fillna(1, inplace = True)
players["SprintSpeed"].fillna(1, inplace = True)
players["Agility"].fillna(1, inplace = True)
players["Reactions"].fillna(1, inplace = True)
players["Balance"].fillna(1, inplace = True)
players["ShotPower"].fillna(1, inplace = True)
players["Jumping"].fillna(1, inplace = True)
players["Stamina"].fillna(1, inplace = True)
players["Strength"].fillna(1, inplace = True)
players["LongShots"].fillna(1, inplace = True)
players["Aggression"].fillna(1, inplace = True)
players["Interceptions"].fillna(1, inplace = True)
players["Positioning"].fillna(1, inplace = True)
players["Vision"].fillna(1, inplace = True)
players["Penalties"].fillna(1, inplace = True)
players["Composure"].fillna(1, inplace = True)
players["Marking"].fillna(1, inplace = True)
players["StandingTackle"].fillna(1, inplace = True)
players["SlidingTackle"].fillna(1, inplace = True)
players["GKDiving"].fillna(1, inplace = True)
players["GKHandling"].fillna(1, inplace = True)
players["GKKicking"].fillna(1, inplace = True)
players["GKPositioning"].fillna(1, inplace = True)
players["GKReflexes"].fillna(1, inplace = True)

# drop 60 NA positions from dataframe
players = players.dropna()

#players.isnull().sum()

# randomly separating the data into a train set and a test set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

x_train, x_test = train_test_split(players, test_size=0.2, random_state=42)

# x_train contains 14517 rows
# problem: x_train is an unbalanced dataset
# number of players per position in x_train
# LS: 170 (1.17%) ST: 1726 (11.89%) RS: 162 (1.16%) LW: 303 (2.09%)
# LF: 13 (0.09%) CF: 59 (0.41%) RF: 12 (0.08%) RW: 293 (2.02%)
# LAM: 16 (0.11%) CAM: 787 (5.42%) RAM: 17 (0.12%) LM: 864 (5.95%)
# LCM: 302 (2.08%) CM: 1112 (7.66%) RCM: 305 (2.10%) RM: 885 (6.10%)
# LWB: 62 (0.43%) LDM: 194 (1.34%) CDM: 764 (5.26%) RDM: 206 (1.42%)
# RWB: 74 (0.50%) LB: 1050 (7.23%) LCB: 520 (3.58%) CB: 1446 (9.96%)
# RCB: 528 (3.64%) RB: 1029 (7.09%) GK: 1618 (11.15%)

# number of players per section in x_train
# attack (LS + ST + ... + RW): 2738 (18.86%)
# midfield (LAM + CAM + ... + RDM): 5452 (38.56%)
# defense (LWB + RWB + LB + LCB + ... + RB): 4709 (32.44%)
# GK: 1618 (11.15%)

# 1. labels have to be numerical
# 2. one-hot encoding has to be used instead of integer encoding, so that the model doesn't assume a natural ordering between categories
y_train, y_test = [], []
sections_train, sections_test = [], []

# first, the categorical data has to be mapped to integer data
for i in range(len(x_train["Position"])):
    y_train.append(positions.index(x_train["Position"].iloc[i]))
    if x_train["Position"].iloc[i] in attacking_positions:
        sections_train.append(0)
    elif x_train["Position"].iloc[i] in midfield_positions:
        sections_train.append(1)
    elif x_train["Position"].iloc[i] in defensive_positions:
        sections_train.append(2)
    elif x_train["Position"].iloc[i] == "GK":
        sections_train.append(3)

# now, the integer data can be one-hot encoded
y_train = keras.utils.to_categorical(y_train)
sections_train = keras.utils.to_categorical(sections_train)

x_train = x_train.drop(["Position"], axis=1)

# normalization is used instead of standardization, because the data is not Gaussian
x_train = keras.utils.normalize(x_train.values, axis=1)

# plot x_train
# plt.style.use('ggplot')
# ax = sns.distplot(x_train[:][0], bins = 52, kde = False, color = 'r')
# plt.show()

for i in range(len(x_test["Position"])):
    y_test.append(positions.index(x_test["Position"].iloc[i]))
    if x_test["Position"].iloc[i] in attacking_positions:
        sections_test.append(0)
    elif x_test["Position"].iloc[i] in midfield_positions:
        sections_test.append(1)
    elif x_test["Position"].iloc[i] in defensive_positions:
        sections_test.append(2)
    elif x_test["Position"].iloc[i] == "GK":
        sections_test.append(3)

y_test = keras.utils.to_categorical(y_test)
sections_test = keras.utils.to_categorical(sections_test)

x_test = x_test.drop(["Position"], axis=1)

x_test = keras.utils.normalize(x_test.values, axis=1)

from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout

model = keras.models.Sequential()

# a simple fully-connected layer, 128 units, relu activation
model.add(Dense(128, kernel_initializer='random_normal', input_dim=len(x_test[0])))
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

# model.add(tf.keras.layers.Flatten()) # a simple fully-connected layer, 128 units, relu activation

# output layer. 27 units for 27 classes (= positions). Softmax for probability distribution
# model.add(Dense(27, kernel_initializer='random_normal'))
# output layer. 4 units for 4 classes (= sections). Softmax for probability distribution
model.add(Dense(4, kernel_initializer='random_normal'))
model.add(BatchNormalization())
model.add(Activation('softmax'))

adam = keras.optimizers.Adam(lr=0.01)

# Adam = RMSProp + Momentum
# categorical_crossentropy should be used for one-hot encoded data
# TODO: USE AUC  OR F1 MEASURE INSTEAD OF ACCURACY, BECAUSE THEY ARE BETTER FOR UNBALANCED DATASETS
model.compile(optimizer=adam,  # Good default optimizer to start with
              loss='categorical_crossentropy',
              # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

section_ints = [section.argmax() for section in sections_train]
# treat 1 instance of an attacker/goalkeeper as 2 instances of a midfielder/defender
class_weights = {0: 1., 1: 2, 2: 2, 3: 1.}

#model.fit(x_train, y_train, epochs=3, batch_size=100)  # train the model
model.fit(x_train, sections_train, epochs=100, batch_size=100, class_weight=class_weights)  # train the model

#val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
val_loss, val_acc = model.evaluate(x_test, sections_test)  # evaluate the out of sample data with model

predictions = model.predict(x_test)

'''
for i in range(len(predictions)):
    predicted_index = np.argmax(predictions[i])
    actual_index = np.argmax(sections_test[i])
    #print("Predicted position: " + positions[predicted_index] + ", Actual position: " + positions[actual_index])
    print("Predicted section: " + sections[predicted_index] + ", Actual section: " + sections[actual_index])
'''