import seaborn as sns
import matplotlib.pyplot as plt
import globals as GLOBALS
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# plot x_train using plt
# plt.style.use('ggplot')
# ax = sns.distplot(x_train[:][0], bins = 52, kde = False, color = 'r')
# plt.show()



def plot_number_of_players_by_position(players):
    sns.set(style="darkgrid")

    plot = sns.countplot(x="Position", data=players, order=players["Position"].value_counts().index)
    figure = plot.get_figure()

    figure.set_size_inches(12.8, 7.2)
    figure.savefig("players_by_position.png")
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
    figure.savefig("players_by_section.png")
    plt.show()

def plot_correlation_matrix():
    pass

def plot_some_attributes(players):
    pass