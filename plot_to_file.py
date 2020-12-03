import matplotlib.pyplot as plt

"""
Writes png file of x and y data to under plots/ folder
"""


def plot_to_file(algorithm_name, sample, force, thetadiff):
    plt.figure()

    plt.plot(sample, force, label="line 1")

    plt.plot(sample, thetadiff, label="line 2")

    plt.title("MPC behaviour ({})".format(algorithm_name))
    plt.xlabel("Sample")
    plt.ylabel("Magnitude")
    plt.savefig("plots/{}_mpc_plot".format(algorithm_name))
    plt.close()
