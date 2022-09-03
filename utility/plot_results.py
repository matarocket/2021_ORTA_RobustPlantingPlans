# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


#PLOT: histogram for the in sample stability (heuristic solver) 

def plot_hist_in_heu(exact):
    pyplot.hist(exact, bins=15, alpha=1, color="gold")
    plt.grid()
    pyplot.xlabel("Objective function [£]")
    pyplot.ylabel("Occurrencies")
    plt.title("In-sample stability - heuristic")
    pyplot.savefig(f"./results/hist_inSample_heu.png")
    pyplot.close()
  


#PLOT: histogram for the in sample stability (exact solver) 

def plot_hist_in_exact(exact):
    pyplot.hist(exact, bins=15, alpha=1, color="dodgerblue")
    plt.grid()
    pyplot.xlabel("Objective function [£]")
    pyplot.ylabel("Occurrencies")
    plt.title("In-sample stability - exact")
    pyplot.savefig(f"./results/hist_inSample_exact.png")
    pyplot.close()



#PLOT: histogram for the out of sample stability (heuristic solver) 

def plot_hist_out_heu(exact):
    pyplot.hist(exact, bins=15, alpha=1, color="limegreen")
    plt.grid()
    pyplot.xlabel("Objective function [£]")
    pyplot.ylabel("Occurrencies")
    plt.title("Out-of-sample stability - heuristic")
    pyplot.savefig(f"./results/hist_outSample_heu.png")
    pyplot.close()



#PLOT: histogram for the out of sample stability (exact solver) 

def plot_hist_out_exact(exact):
    pyplot.hist(exact, bins=15, alpha=1, color="red")
    plt.grid()
    pyplot.xlabel("Objective function [£]")
    pyplot.ylabel("Occurrencies")
    plt.title("Out-of-sample stability - exact")
    pyplot.savefig(f"./results/hist_outSample_exact.png")
    pyplot.close()



#PLOT: Impact of w on the Profit and its variability (individual plots)

def plot_w_comparison(w_vector, mean_w_exact, stddev_w_norm):

    plt.plot(w_vector, mean_w_exact, color='dodgerblue')
    plt.xlabel("w")
    plt.ylabel("Normalized Expected Profit [£]")
    plt.title("Mean results by varying w")
    plt.grid()
    pyplot.savefig(f"./results/w_comparision_profit.png")
    pyplot.close()


    plt.plot(w_vector, stddev_w_norm, color='limegreen')
    plt.xlabel("w")
    plt.ylabel("Profit Normalized Standard deviation [£]")
    plt.title("Mean results by varying w")
    plt.grid()
    pyplot.savefig(f"./results/w_comparision_std.png")
    pyplot.close()



#PLOT: Impact of w on the Profit and its variability (together)

def plot_w_comparison_together(w_vector, mean_w_exact, stddev_w_norm):
    plt.plot(w_vector, mean_w_exact, color='dodgerblue', label="mean profit")
    plt.plot(w_vector, mean_w_exact+stddev_w_norm*0.00001, color='limegreen', linestyle="--", label="standard deviation")
    plt.plot(w_vector, mean_w_exact-stddev_w_norm*0.00001, color='limegreen', linestyle="--")
    plt.xlabel("w")
    plt.legend()
    plt.ylabel("Normalized Expected Profit [£]")
    plt.title("Mean results by varying w")
    plt.grid()
    pyplot.savefig(f"./results/w_together.png")
    pyplot.close()



#PLOT: for the observation of the behaviours of the exact solver and the heuristic 
#      when the number of scenarios is increased 

def plot_comparison_compTimes(N, comp_G, comp_Heu):
    plt.plot(range(1,1+N), comp_G, label="exact", color="dodgerblue")
    plt.plot(range(1,1+N),comp_Heu, label="heuristic", color="gold")
    plt.xlabel("N° of scenarios")
    plt.ylabel("Computational time [s]")
    plt.grid()
    plt.title("Computational time over the number of scenarios")
    plt.legend()
    pyplot.savefig(f"./results/comp_time.png")
    pyplot.close()



#PLOT: for the observation of the behaviours of the exact solver and the heuristic 
#      when the dimensionality of the crops is enlarged 

def plot_comparison_compTimes_crops(N, comp_G, comp_Heu):
    plt.plot(range(3,3+N), comp_G, label="exact", color="dodgerblue")
    plt.plot(range(3,3+N),comp_Heu, label="heuristic", color="gold")
    plt.xlabel("Crop dimensionality [n x n x n]")
    plt.ylabel("Computational time [s]")
    plt.grid()
    plt.title("Computational time over the dimensionality settings")
    plt.legend()
    pyplot.savefig(f"./results/comp_time_crops.png")
    pyplot.close()



# CONTROL print in terminal: to show the comparison between the performance of 
#                            Gurobi compared to the heuristic solution

def control_print(of_exact, sol_exact, comp_time_exact, of_heu, sol_heu, comp_time_first, comp_time_second):
    
    print("\n\n\n>> Profit exact solver :  ", of_exact," <<\n")
    print(">> Sowing plan (A_i) :  ", sol_exact," <<\n")
    print(">> Total computational time :  ", comp_time_exact," <<\n")

    print("\n\n\n>> Profit heuristic solver :  ", of_heu," <<\n")
    print(">> Heuristic sowing plan (A_i) :  ", sol_heu," <<\n")
    print(">> First stage computational time for heuristic :  ", comp_time_first," <<")
    print(">> Second stage computational time for heuristic :  ", comp_time_second," <<")
    print(">> Total computational time for heuristic :  ", comp_time_second+comp_time_first," <<\n\n\n")

    print(">> Percentual difference :  ", 100*(of_exact - of_heu)/of_exact, " <<")
    
    return



