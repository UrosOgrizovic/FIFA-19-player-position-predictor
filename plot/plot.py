import seaborn as sns
import matplotlib.pyplot as plt

# plot x_train using plt
# plt.style.use('ggplot')
# ax = sns.distplot(x_train[:][0], bins = 52, kde = False, color = 'r')
# plt.show()

def plot_number_of_players_by_position(players):
    number_of_players_by_position = players["Position"].value_counts()
    sns.set(style="darkgrid")

    plot = sns.countplot(x="Position", data=players, order=players["Position"].value_counts().index)
    figure = plot.get_figure()

    figure.set_size_inches(12.8, 7.2)
    figure.savefig("players_by_position.png")
    plt.show()


def plot_number_of_players_by_section():
    pass

def plot_correlation_matrix():
    pass