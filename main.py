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
import model.Random_Forest as RF

# number of positions = 27
positions = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM",
             "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM",
             "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB", "GK"]

sections = ["ATT", "MID", "DEF", "GK"]

attacking_positions = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW"]
midfield_positions = ["LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LDM", "CDM", "RDM"]
defensive_positions = ["LWB", "RWB", "LB", "LCB", "CB", "RCB", "RB"]

def main():
    data = DATA.read_data()

    #PLOT.plot_correlation_matrix(data)
    players = DATA.make_players(data, "1")

    # PLOT.plot_number_of_players_by_position(players)
    # PLOT.plot_number_of_players_by_section(players)
    # PLOT.plot_some_attributes(players, "ShotPower")


    # randomly splitting data into train/test sets in 80/20 ratio
    x_train, x_test = train_test_split(players, test_size=0.2)

    # 1. labels have to be numerical
    # 2. one-hot encoding has to be used instead of integer encoding, so that the model doesn't assume a natural ordering between categories
    positions_train, positions_test = [], []
    sections_train, sections_test = [], []

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

    DATA.populate_positions(x_test, positions_test)
    DATA.populate_sections(x_test, sections_test)

    positions_test = keras.utils.to_categorical(positions_test)
    sections_test = keras.utils.to_categorical(sections_test)

    x_test = x_test.drop(["Position"], axis=1)
    x_test = keras.utils.normalize(x_test.values, axis=1)

    ###

    # do_ANN(x_train, sections_train, x_test, sections_test, "Section")
    do_random_forest(x_train, sections_train, x_test, sections_test, "Section")

def do_ANN(x_train, y_train, x_test, y_test, position_or_section):
    if position_or_section.upper() == "POSITION":
        model = ANN.make_NN(x_train, 27)
    elif position_or_section.upper() == "SECTION":
        model = ANN.make_NN(x_train, 4)
    else:
        raise ValueError("Invalid value for position_or_section parameter")

    predictions = model.predict(x_test)
    # tried number of epochs: 10, 100, 100 - none of them gave an accuracy of over 89%
    # tried batch sizes: 10, 20, 50, 100, 200, 500 - none of them gave an accuracy of over 89%
    train_model(model, x_train, y_train, 100, 100, position_or_section)
    # DATA.display_predictions(predictions, y_test, position_or_section)


def do_random_forest(x_train, y_train, x_test, y_test, position_or_section):
    rf_model = RF.random_forest()
    rf_model.fit(x_train, y_train)
    print("-" * 50 + " Random forest " + "-" * 50)
    print("Train set accuracy: ", rf_model.score(x_train, y_train))
    # the score could be inaccurate, as the model might just be overfitting
    # that is why the out-of-bag score is used
    print("Train set out-of-bag accuracy: ", rf_model.oob_score_)
    print("Test set accuracy: ", rf_model.score(x_test, y_test))
    rf_predictions = rf_model.predict(x_test)
    # DATA.display_predictions(rf_predictions, y_test, position_or_section)
    print("-" * 100)


def train_model(model, x_train, y_train, num_of_epochs, batch_size, position_or_section):
    class_weights = None
    if position_or_section.upper() == "POSITION":
        position_or_section = "Position"
    else:
        position_or_section = "Section"
        class_weights = {0: 1., 1: 2, 2: 2, 3: 1.}
    dashes = "-"*50
    print(dashes + " " + position_or_section + " model training " + dashes)
    ANN.train_NN(model, x_train, y_train, class_weights, num_of_epochs, batch_size)
    print(dashes*2)

if __name__ == "__main__":
    main()