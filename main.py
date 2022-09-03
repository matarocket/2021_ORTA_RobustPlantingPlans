#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
from simulator.instance import Instance
from simulator.tester import Tester
from solver.robustPlantingPlan import RobustPlantingPlanSolver
from heuristic.firstStageHeuristicALNS import Heuristic
from solver.sampler import Sampler
import utility.plot_results as pr
import utility.comparisons as cp



if __name__ == '__main__':

    # %% INITIALIZATION phase 

    log_name = "./logs/main.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )


    fp = open("./etc/sim_setting.json", 'r')
    sim_setting = json.load(fp)
    fp.close()


    sam = Sampler()
    test = Tester()
    prb = RobustPlantingPlanSolver()


    inst = Instance(sim_setting)
    dict_data = inst.get_data()
    print(dict_data)
   
    # Prob scenarios
    prob_s = sam.sample_stoch(inst)


    # # %% GUROBI solution

    # of_exact, sol_exact, comp_time_exact, model = prb.solve(
    #     dict_data,
    #     prob_s=prob_s
    # )


    # # %% HEURISTIC solution
     
    # of_heu, sol_heu, comp_time_second, comp_time_first = Heuristic.solve(dict_data, prob_s)


    # # %% PERFORMANCE comparison 

    # pr.control_print(of_exact, sol_exact, comp_time_exact, of_heu, sol_heu, comp_time_first, comp_time_second)
   

    # %% STABILITY

    # #IN SAMPLE STABILITY
    # N_scen_tot=8 #number of scenarios to be used 
    # N_repetitions=100 #number of iterations to be performed 
    # test.in_sample_stability(N_scen_tot, sam, prb,sim_setting, N_repetitions)


    # #OUT OF SAMPLE STABILITY
    # N_scen_tot=10 #number of scenarios for the training part 
    # N_repetitions=100 #number of scenarios for the testing phase  
    # test.out_of_sample_stability(N_scen_tot, sam, prb,sim_setting, N_repetitions)
    

    # %% COMPARISON HEURISTIC - GUROBI IN TERMS OF TIME 


    # #>> 1: evaluation of computational time over the changing number of scenarios 
    # N = 50 #number of scenarios to be tested 
    # cp.scenario_increasing_comparison(N, sam, prb)


    #>> 2: evaluation of computational time over the changing number of crops 
    # N = 15 #number of dimensionalities to be explored - it starts from 3 and does N more 
    # cp.dimensionality_increasing_comparison(N, sam, prb)

    # %%COMPARISON of GUROBI profits by changing risk term (w)
    N = 15 #number of considered problems 
    N_samples=10 #number of considered samples 
    cp.w_changing(sim_setting, sam, prb, N, N_samples)
  

  
    

