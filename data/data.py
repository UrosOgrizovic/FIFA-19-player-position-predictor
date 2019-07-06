import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

positions = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW", "LAM",
             "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LWB", "LDM",
             "CDM", "RDM", "RWB", "LB", "LCB", "CB", "RCB", "RB", "GK"]

sections= ["ATT", "MID", "DEF", "GK"]

attacking_positions = ["LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW"]
midfield_positions = ["LAM", "CAM", "RAM", "LM", "LCM", "CM", "RCM", "RM", "LDM", "CDM", "RDM"]
defensive_positions = ["LWB", "RWB", "LB", "LCB", "CB", "RCB", "RB"]

def read_data():
    return pd.read_csv('data/data.csv')

def make_players(data):
    """
    1. feature selection
    2. replacing null values
    :param data:
    :return: players
    """
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
    #TODO: try replacing null values with mean/median instead

    for col in players:
        if col != "Position":
            players[col].fillna(1, inplace=True)

    # drop 60 NA positions from dataframe
    players = players.dropna()

    return players

def populate_positions(df, positions_set):
    """
    adds ST/CB/LM/... to position set
    :param df:
    :param positions_set:
    :return:
    """
    for i in range(len(df["Position"])):
        positions_set.append(positions.index(df["Position"].iloc[i]))

# The reasoning behind diving the players into sections:
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
# Although the number of players by section is also unbalanced,
# it is less so unbalanced than the number of players by position
def populate_sections(df, sections):
    """
    adds ATT/MID/DEF/GK to sections
    :param df:
    :param sections:
    :return:
    """
    for i in range(len(df["Position"])):
        if df["Position"].iloc[i] in attacking_positions:
            sections.append(0)
        elif df["Position"].iloc[i] in midfield_positions:
            sections.append(1)
        elif df["Position"].iloc[i] in defensive_positions:
            sections.append(2)
        elif df["Position"].iloc[i] == "GK":
            sections.append(3)

def display_predictions(predictions, test_set, position_or_section):
    """
    displays predicted position/section vs actual position/section for each player from predictions
    :param predictions:
    :param test_set:
    :param position_or_section:
    :return:
    """
    for i in range(len(predictions)):
        predicted_index = np.argmax(predictions[i])
        actual_index = np.argmax(test_set[i])
        if position_or_section.upper() == "POSITION":
            print("Predicted position: " + positions[predicted_index] + ", Actual position: " + positions[actual_index])
        else:
            print("Predicted section: " + sections[predicted_index] + ", Actual section: " + sections[actual_index])

