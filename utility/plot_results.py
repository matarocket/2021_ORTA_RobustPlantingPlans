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

def plot_comparison_compTimes(N, comp_G, comp_Heu):
    plt.plot(range(1,1+N), comp_G, label="exact", color="dodgerblue")
    plt.plot(range(1,1+N),comp_Heu, label="heuristic", color="gold")
    plt.xlabel("N° of scenarios")
    plt.ylabel("Computational time [s]")
    plt.grid()
    plt.title("Computational time over the number of scenarios")
    plt.legend()
    plt.show()
    #pyplot.savefig(f"C:\\Users\\Giulia\\Desktop\\PoliTO\\Operational research\\2021_ORTA_RobustPlantingPlans\\results\\comp_time.png")
    #pyplot.close()


def plot_comparison_compTimes_crops(N, comp_G, comp_Heu):
    plt.plot(range(3,1+N), comp_G, label="exact", color="dodgerblue")
    plt.plot(range(3,1+N),comp_Heu, label="heuristic", color="gold")
    plt.xlabel("Crop dimensionality [n x n x n]")
    plt.ylabel("Computational time [s]")
    plt.grid()
    plt.title("Computational time over the dimensionality settings")
    plt.legend()
    plt.show()
    #pyplot.savefig(f"C:\\Users\\Giulia\\Desktop\\PoliTO\\Operational research\\2021_ORTA_RobustPlantingPlans\\results\\comp_time_crops.png")
    #pyplot.close()

def mean_std_plot(n_scenarios_vector, in_sample_res_std, in_sample_res_mean):
    plt.errorbar(n_scenarios_vector, in_sample_res_mean, in_sample_res_std,capsize=5, linestyle='None', marker='o', ecolor="lightcoral", c="dodgerblue")
    plt.xlabel("N° scenarios")
    plt.ylabel("Mean and std of profits")
    plt.grid()
    plt.title("Mean and std over different N° of scenarios")
    plt.legend()
    #plt.show()
    pyplot.savefig(f"C:\\Users\\Giulia\\Desktop\\PoliTO\\Operational research\\2021_ORTA_RobustPlantingPlans\\results\\mean_std.png")
    pyplot.close()

def mean_std_plot_out(x, in_sample_res_std, in_sample_res_mean):
    plt.errorbar(x, in_sample_res_mean, in_sample_res_std,capsize=5, linestyle='None', marker='o', ecolor="lightcoral", c="dodgerblue")
    plt.xlabel("N° scenarios")
    plt.ylabel("Mean and std of profits")
    plt.grid()
    plt.title("Mean and std over different N° of scenarios")
    plt.legend()
    #plt.show()
    pyplot.savefig(f"C:\\Users\\Giulia\\Desktop\\PoliTO\\Operational research\\2021_ORTA_RobustPlantingPlans\\results\\mean_std_out.png")
    pyplot.close()