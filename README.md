# Robust Planting Plan

In order to use this set of scripts you need the following software components:

1. Python 3.6
1. gurobi

## Python

Linux users should have python already installed in their PC. For other users I strongly recomment to install **Anaconda** and to use its terminal for installing new packages. You can find details [here](https://www.anaconda.com/distribution/) and [there](https://www.anaconda.com/distribution/#download-section). 


## gurobi
gurobi is a commercial software. See [here](https://www.gurobi.com/)


## Python packages:
Probably you will need to install several packages (e.g., pulp, numpy, etc). In linux my suggestion is to use pip
~~~
pip3 install <package name>
~~~
e.g., 
~~~
pip3 install pulp
~~~
For windows is suggest to use conda.
~~~
conda install -c conda-forge pulp 
~~~


## Repository Structure:
Run the code by writing in the terminal
```
python3 main.py
```
the main script contains different blocks of code that are used to produce different performance results on the proposed solution. The structure of the folder and scripts is the following:
```
project
│   README.md
│   main.py : Main file with all the simulations done on the project    
│
└─── Documentation : Contais an article and the support files with the explanation for the whole project
│   
└─── etc
|       sim_setting.json : File continaing the hyperparameters for the problem to solve
|
└─── heuristics : Contains the scripts needed to solve the problem through heuristics
|       firstStageHeuristicALNS.py : First stage variables through ALNS
|       secondStageHeuristicGurobi.py : Second stage variables through Gurobi with single scenario
|
└─── logs : Folder on the .gitignore where the output logs for the optimizations are saved
|
└─── results : Folder where the output plots for the optimizations are saved
|
└─── simulator : Contains the scripts needed for the problem generation
|       instance.py : Script for the generation fo new random instances
|       tester.py : Script for the testing of the in-sample and out-sample stabilities
|
└─── solver : Contains the scripts needed to solve the problem with Gurobi
|       robustPlantingPlan.py : Solves the problem with Gurobi
|       sampler.py : Solves the problem with Gurobi
|
└─── utility : Different scripts needed for generating plots and results

```

## Robust Planting Plan Problem

The object of the project is to set up a Planting Plan with Risk Management, considering a problem in which in a first stage, the planting plan is constructed, and a second stage will set up the harvesting plan for a given number of possible scenarios.

We consider the following Object Function (O.F.) to be maxized:
$$
O.F.=(weighted)\cdot ExpecteProfit-RiskTerm=(1-\omega)\cdot E(Profit_{s})-\omega\cdot E(|Profit_{s}-E(Profit_{s})|)
$$
where the profit is defined as:
$$
E(Profit_{s})=\sum_{s}prob_{s}\cdot\left(\sum_{j,k}s_{sj}\cdot S_{sjk}+\sum_{m,j,k}f_{mj}\cdot F_{sjmk}-\sum_{i,j}c_{sij}\cdot H_{sij}+c^{+}\cdot L^{+}-c^{-}\cdot L^{-}-c'\cdot\sum_{i}A_{i}-\sum_{m,j}p_{smj}\cdot P_{smj}\right)
$$

The maximization must be done respecting a series of constraints that are linked to the market and the nature of the plants. They are listed hereunder:


### Marketing Constraint 
Expresses the fact that the weekly harvest is sold either to customers or on the open market:
$$
\sum_{i}y_{sijk}\cdot H_{sij}-S_{sjk}-\sum_{m}F_{sjmk}=0
$$

### Demand Constraint
The quantity of sprouts sold is given by the sum in the demand the customer and shortage:
$$
\sum_{k}F_{sjmk}=P_{smj}+d_{mj}
$$

### Sell on Open Market 
Imposes a limit on the amount of sprouts that can be sold on free-trade option:
$$
\sum_{k}S_{sjk}\leq0.25\cdot\sum_{m}d_{mj}
$$

### Land Use Constraints  
Imposes a limit on the amount of sprouts that can be sold on free-trade option:
$$
\sum_{i}A_{i}=a+L^{-}-L^{+}
$$
Another constraint imposes the area of each crop planted matches the harvested area:
$$
A_{i}=\sum_{j}H_{sij}
$$

### Individual Variety Limit   
Sets limits on the amount of demand of crops that can be susceptible to each kind of plant illness:
$$
\sum_{i\in I_{v}}A_{i}\leq0.4\cdot\sum_{i}A_{i}
$$

### Individual Crop Limit
Sets limits on the amount of area assigned to each crop:
$$
A_{i}\leq0.2\cdot\sum_{i}A_{i}
$$






