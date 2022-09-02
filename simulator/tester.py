# -*- coding: utf-8 -*-
import os
import time
import logging
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from simulator.instance import Instance
from heuristic.firstStageHeuristicALNS import Heuristic
import heuristic.secondStageHeuristicGurobi as ss


class Tester():
    def __init__(self):
        pass


    def in_sample_stability(self, N_scen_tot, sam, problem,sim_setting, n_repetitions):
        ans=[]
        ans_heu=[]
        
        print(">>> IN SAMPLE stability GUROBI <<<")
    
        for i in range(n_repetitions):

            sim_setting["n_scenarios"] = N_scen_tot

            inst = Instance(sim_setting)
            dict_data = inst.get_data()
            prob_s = sam.sample_stoch(inst)
            inst.prob_s = prob_s

            print(">>>>> scenario N° ", N_scen_tot , " it. N° ", i)

            of, _, _, _ = problem.solve(
            dict_data,
            prob_s)

            of_heu, _, _, _ = Heuristic.solve(dict_data, prob_s)

            ans.append(of)
            ans_heu.append(of_heu)


        return ans, ans_heu


  


    def out_of_sample_stability(self, N_scen_tot, sam, problem,sim_setting, n_repetitions):
        
        print(">>> OUT OF SAMPLE stability <<<")

        ans=[]
        ans_heu=[]

        
        sim_setting["n_scenarios"] = N_scen_tot
        inst = Instance(sim_setting)
        dict_data = inst.get_data()
        prob_s = sam.sample_stoch(inst)
        inst.prob_s = prob_s

        _, sol, _, model = problem.solve(
            dict_data,
            prob_s
            )

        _, sol_heu, _, _ = Heuristic.solve(dict_data, prob_s)

        grb_var = model.getVarByName("Lminus[0]") 
        L_minus = grb_var.X
        grb_var = model.getVarByName("Lplus[0]") 
        L_plus = grb_var.X

    

        sim_setting["n_scenarios"] = n_repetitions
        inst1 = Instance(sim_setting)
        dict_data1 = inst1.get_data()
        prob_s1 = sam.sample_stoch(inst1)
        inst1.prob_s = prob_s1


    
        secondStage=ss.SecondStageSolver()

        for k in range(n_repetitions):
            print(">>>>> exact it. N° ", k)
            of_heu, _, _ = secondStage.solve(dict_data1, k, sol, L_plus, L_minus)
            ans.append(of_heu)
        
   
        for k in range(n_repetitions):
            print(">>>>> heu it. N° ", k)
            of_heu, _, _ = secondStage.solve(dict_data1, k, sol_heu,  dict_data["a"]-np.sum(sol_heu), 0)
            ans_heu.append(of_heu)

        
        

        return ans, ans_heu

        


