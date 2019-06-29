# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("input"))

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
# 1. labels have to be numerical
# 2. one-hot encoding has to be used instead of integer encoding, so that the model doesn't assume a natural ordering between categories
y_train, y_test = [], []

# first, the categorical data has to be mapped to integer data
for i in range(len(x_train["Position"])):
    y_train.append(positions.index(x_train["Position"].iloc[i]))

# now, the integer data can be one-hot encoded
y_train = keras.utils.to_categorical(y_train)

x_train = x_train.drop(["Position"], axis=1)

# standardization is required because skill moves range from 1 to 5, whereas everything else ranges from 1 to 99
# this also converts x_train from a pandas DataFrame to a numpy array
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

# min_max_scaler = preprocessing.MinMaxScaler()
# x_train = min_max_scaler.fit_transform(x_train.values)
# x_train = pd.DataFrame(x_train)

# x_train = keras.utils.normalize(x_train, axis=1)

for i in range(len(x_test["Position"])):
    y_test.append(positions.index(x_test["Position"].iloc[i]))

y_test = keras.utils.to_categorical(y_test)

x_test = x_test.drop(["Position"], axis=1)
sc = StandardScaler()
x_test = sc.fit_transform(x_test)
# x_test = keras.utils.normalize(x_test, axis=1)

model = keras.models.Sequential()

model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu, kernel_initializer='random_normal'))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(4, activation=tf.nn.relu, kernel_initializer='random_normal'))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(27, activation=tf.nn.softmax, kernel_initializer='random_normal'))  # our output layer. 27 units for 27 classes (= positions). Softmax for probability distribution

# Adam = RMSProp + Momentum
# categorical_crossentropy should be used for one-hot encoded data
model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

model.fit(x_train, y_train, epochs=3, batch_size=20)  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model

predictions = model.predict(x_test)
#print(np.argmax(predictions[0]))
print(positions[np.argmax(predictions[0])])

print(positions[np.argmax(y_test[0])])
