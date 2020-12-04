import matplotlib.pyplot as plt

"""
Writes png file of x and y data to under plots/ folder
"""


def plot_to_file(algorithm_name, x, y):
    plt.figure()
    plt.plot(x, y)
    plt.title("Rewards over time ({})".format(algorithm_name))
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("plots/{}_epoc_reward".format(algorithm_name))
    plt.close()
