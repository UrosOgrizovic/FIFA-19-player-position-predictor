import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import globals as GLOBALS
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def plot_number_of_players_by_position(players):
    sns.set(style="darkgrid")

    plot = sns.countplot(x="Position", data=players, order=players["Position"].value_counts().index)

    figure = plot.get_figure()

    figure.set_size_inches(12.8, 7.2)
    figure.savefig("plot/players_by_position.png")
    plt.show()


def plot_number_of_players_by_section(players):
    num_of_players_by_section = {"ATT": 0, "MID": 0, "DEF": 0, "GK": 0}

    for i in range(len(players["Position"])):
        if players["Position"].iloc[i] in GLOBALS.attacking_positions:
            num_of_players_by_section["ATT"] += 1
        elif players["Position"].iloc[i] in GLOBALS.midfield_positions:
            num_of_players_by_section["MID"] += 1
        elif players["Position"].iloc[i] in GLOBALS.defensive_positions:
            num_of_players_by_section["DEF"] += 1
        elif players["Position"].iloc[i] == "GK":
            num_of_players_by_section["GK"] += 1


    df = pd.DataFrame(num_of_players_by_section.items(), columns=["Section", "Number of players"])
    df = df.sort_values(["Number of players"], ascending=False).reset_index(drop=True)
    sns.set(style="darkgrid")

    plot = sns.barplot(x="Section", y="Number of players", data=df)
    figure = plot.get_figure()
    figure.set_size_inches(12.8, 7.2)
    figure.savefig("plot/players_by_section.png")
    plt.show()

def plot_correlation_matrix(players):
    """
    1. plots correlation matrix, to see which values can be removed
    2. saves correlation matrix to "correlation_matrix.png"
    :param players:
    :return:
    """
    # Compute the correlation matrix
    corr = players.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    figure, ax = plt.subplots(figsize=(12.8, 7.2))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    figure.savefig("plot/" + "correlation_matrix.png")
    plt.show()

def plot_some_attributes(players, attribute_name):
    # plot x_train using plt
    plt.style.use('ggplot')
    plot = sns.distplot(players[attribute_name])
    figure = plot.get_figure()
    figure.set_size_inches(12.8, 7.2)
    figure.savefig("plot/" + attribute_name.lower() + ".png")
    plt.show()