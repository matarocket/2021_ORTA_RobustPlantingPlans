# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


def plot_comparison_hist(values, labels, colors, x_label, y_label):
    for i, item in enumerate(values):
        pyplot.hist(item, color=colors[i], bins=100, alpha=0.5, label=labels[i])
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.legend(loc='upper left')
    pyplot.savefig(f"./results/hist_profit.png")
    pyplot.close()

def plot_w_comparison(w_vector, mean_w_exact, stddev_w_norm):
    #Plots
    plt.plot(w_vector, mean_w_exact, color='blue')
    plt.xlabel("w")
    plt.ylabel("Normalized Expected Profit [eur]")
    plt.title("Mean results by varying w")
    plt.grid()
    pyplot.savefig(f"./results/w_comparision_profit.png")
    pyplot.close()
    #plt.show()
    plt.plot(w_vector, stddev_w_norm, color='blue')
    plt.xlabel("w")
    plt.ylabel("Profit Normalized Standard deviation [eur]")
    plt.title("Mean results by varying w")
    plt.grid()
    pyplot.savefig(f"./results/w_comparision_std.png")
    pyplot.close()
    #plt.show()