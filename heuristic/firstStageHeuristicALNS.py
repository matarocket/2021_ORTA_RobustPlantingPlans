import copy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import heuristic.secondStageHeuristicGurobi as heu_second

from alns import ALNS, State
from alns.accept import *
from alns.stop import *
from alns.weights import *
import time 

SEED = 42

np.random.seed(SEED)


n = 100
p = np.random.randint(1, 100, size=n)
w = np.random.randint(10, 50, size=n)
W = 1_000


# Percentage of items to remove in each iteration
DESTROY_RATE = .1
MAX_ITERATIONS = 100

class Heuristic():

    def __init__(self,dict_data):
        self.dict_data = dict_data

    def collapse_prob(mat, prob_s):
        e_s = np.copy(mat)
        for s in range(len(prob_s)):
            e_s[s] = prob_s[s]*e_s[s]
        e_s = np.sum(e_s, axis=0)
        return e_s

    def weekly_demand_matrix(dict_data, y_ijk):
        customers = range(dict_data["customers"])
        weeks = range(dict_data["weeks"])
        d_jk = np.zeros((dict_data["weeks"],dict_data["bands"]))
        for m in customers:
            bands = dict_data["Km"][m] #Array of prefered bands
            for j in weeks:
                demand = dict_data["d_mj"][m][j]
                crops_production = y_ijk[:, j, bands] #Week is an int, band is an array
                crops_production_max = np.amax(crops_production,0) #For each crop use the most productive one
                crops_production_idx = np.argmax(crops_production, axis=0) #Best bands
                best_band_idx = np.argmax(crops_production_max, axis=0)
                crops_production_idx = crops_production_idx[best_band_idx]
                best_band = bands[best_band_idx]
                d_jk[j,best_band] += demand     
        return d_jk

    def solve(dict_data, prob_s):

        crit = HillClimbing()

        weights = SimpleWeights(scores=[5, 2, 1, 0.5],
                        num_destroy=1,
                        num_repair=1,
                        op_decay=0.8)
        
        occupation_matr = np.full((dict_data["weeks"],dict_data["bands"]), -1)
        initial_sol = np.zeros(dict_data["crops"])
        initial_sol[0] = dict_data["a"]

        sowingState = SowingState(initial_sol,dict_data,prob_s, occupation_matr)
        start=time.time()
        alns = make_alns()
        alns.iterate(sowingState, weights, crit, MaxIterations(MAX_ITERATIONS))
        end=time.time()
        comp_time_first = end-start
        # print("Best objective: ",sowingState.best_sol)
        # print("Best a_i: ",sowingState.best_a_i)
        
        '''_, ax = plt.subplots(figsize=(12, 6))
        res.plot_objectives(ax=ax, lw=2)
        plt.show()'''
        
        heu1 = heu_second.SecondStageSolver()
        
        scenarios=range(dict_data["scenarios"])
        start=time.time()
        profit=0

        for s in scenarios:
            of_heu, sol_heu, comp_time = heu1.solve(dict_data,s, sowingState.best_a_i, dict_data["a"]-sowingState.best_sol,0)
            profit += of_heu*prob_s[s]

        end=time.time()
        comp_time_second=end-start

        return of_heu, sol_heu, comp_time_second, comp_time_first


class SowingState(State):
    """
    Solution class for the 0/1 knapsack problem. It stores the current
    solution as a vector of binary variables, one for each item.
    """

    def __init__(self, a_i, dict_data, prob_s, occupation_matr):
        self.a_i = a_i
        self.dict_data = dict_data
        self.occupation_matr = occupation_matr
        self.best_sol = np.sum(a_i)
        self.best_a_i = a_i
        self.iteration_n = 0
        self.y_ijk = Heuristic.collapse_prob(self.dict_data["y_sijk"], prob_s)
        self.d_jk = Heuristic.weekly_demand_matrix(self.dict_data,self.y_ijk)

    def objective(self):
        obj = np.sum(self.a_i)
        best_solution(self,obj)
        return obj


#def to_destroy(state: KnapsackState) -> int:
    #return int(destroy_rate * state.x.sum())

def random_remove(sowingState, rnd_state):
    # state = copy.deepcopy(state)

    # to_remove = rnd_state.choice(np.arange(n),
    #                              size=to_destroy(state),
    #                              p=state.x / state.x.sum())

    # state.x[to_remove] = 0

    return sowingState
#def random_repair(state: KnapsackState, rnd_state):
    # unselected = np.argwhere(state.x == 0)
    # rnd_state.shuffle(unselected)

    # while True:
    #     can_insert = w[unselected] <= W - state.weight()
    #     unselected = unselected[can_insert]

    #     if len(unselected) != 0:
    #         insert, unselected = unselected[0], unselected[1:]
    #         state.x[insert] = 1
    #     else:
    #         return state

def make_alns() -> ALNS:
    rnd_state = np.random.RandomState(SEED)
    alns = ALNS(rnd_state)
    #alns = ALNS()
    alns.add_destroy_operator(remove)

    alns.add_repair_operator(repair)

    return alns

# Terrible - but simple - first solution, where only the first item is
# selected.


def repair(sowingState, rnd_state):

    #Compute needed area
    for j in range(sowingState.dict_data["weeks"]):
        for k in range(sowingState.dict_data["bands"]):
            if(sowingState.occupation_matr[j,k]==-1):
                occupied = True
                while(occupied == True):
                    crop = np.random.randint(0,sowingState.dict_data["crops"])
                    if(sowingState.a_i[crop]==0):
                        occupied = False
                y_aux = sowingState.y_ijk[crop, j, k]
                needed_area = sowingState.d_jk[j][k]/y_aux
                
                #Land limit constraint
                av_area = sowingState.dict_data["a"] - np.sum(sowingState.a_i) #Initial available area TODO: aggiungere L_minus
        
                #Individual variety limit
                aux=[]
                for i in range(sowingState.dict_data["crops"]):
                        if (sowingState.dict_data["Ai_dict"][i]["Variety"] == sowingState.dict_data["Ai_dict"][crop]["Variety"]):
                            aux.append(i)
                v_land = np.sum(sowingState.a_i[aux])
                av_area = np.min([av_area, 0.4*sowingState.dict_data["a"] - v_land])
        
                #Individual crop limit
                av_area = np.min([av_area, 0.2*sowingState.dict_data["a"]])

                sowingState.a_i[crop] = min(av_area,needed_area)
                sowingState.occupation_matr[j,k]=crop

    return sowingState


def remove(sowingState, rnd_state):

    n_to_remove = round((sowingState.dict_data["weeks"]*sowingState.dict_data["bands"])*DESTROY_RATE)
    indexes = np.argpartition(sowingState.a_i, -n_to_remove)[-n_to_remove:]
    for i in indexes:
        sowingState.a_i[i] = 0
        j,k=np.where(sowingState.occupation_matr == i)
        sowingState.occupation_matr[j,k] = -1

    return sowingState

def best_solution(self,obj):

    if (obj < self.best_sol) and (np.count_nonzero(self.a_i) == np.count_nonzero(self.d_jk)):
        self.best_a_i = copy.copy(self.a_i)
        self.best_sol = obj        
    return