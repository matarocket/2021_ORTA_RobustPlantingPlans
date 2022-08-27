#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import logging
import numpy as np
from simulator.instance import Instance
from simulator.tester import Tester
#from solver.simpleKnapsack import SimpleKnapsack
from solver.robustPlantingPlan import RobustPlantingPlanSolver
#from solver.experiment import RobustPlantingPlanSolver
# from heuristic.heuristicTwoStage import SimpleHeu
# import heuristic.simpleHeu
from heuristic.firstStageHeuristicALNS import Heuristic
from solver.sampler import Sampler
from utility.plot_results import plot_comparison_hist

np.random.seed(0)

if __name__ == '__main__':
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

    inst = Instance(sim_setting)
    dict_data = inst.get_data()
    print(dict_data)
    
    # # Reward generation
    # n_scenarios = 5
    # reward = sam.sample_stoch(
    #     inst,
    #     n_scenarios=n_scenarios
    # )
    
    # Prob scenarios
    prob_s = sam.sample_stoch(inst)

    # mean_reward = sam.sample_ev(
    #     inst,
    #     n_scenarios=n_scenarios
    # )
    # print(mean_reward)

    prb = RobustPlantingPlanSolver()
    of_exact, sol_exact, comp_time_exact, model = prb.solve(
        dict_data,
        prob_s=prob_s
        #verbose=True
    )

    of_heu, sol_heu, comp_time_second, comp_time_first = Heuristic.solve(dict_data, prob_s)

    print("\n\n\n>> Profit exact solver :  ", of_exact," <<\n")
    print(">> Sowing plan (A_i) :  ", sol_exact," <<\n")
    print(">> Total computational time :  ", comp_time_exact," <<\n")

    print("\n\n\n>> Profit heuristic solver :  ", of_heu," <<\n")
    print(">> Heuristic sowing plan (A_i) :  ", sol_heu," <<\n")
    print(">> First stage computational time for heuristic :  ", comp_time_first," <<")
    print(">> Second stage computational time for heuristic :  ", comp_time_second," <<")
    print(">> Total computational time for heuristic :  ", comp_time_second+comp_time_first," <<\n\n\n")

    print(">> Percentual difference :  ", 100*(of_exact - of_heu)/of_exact, " <<")
    
    
    # w_vector = np.linspace(0,1,10)
    # N = 10
    # res_w = []
    
    # idx_w = 0
    # for w in w_vector:
    #     res_w.append([])
    #     for n in range(N):   
    #         # Prob scenarios
    #         prob_s = sam.sample_stoch(inst)
    #         dict_data = inst.get_data()
    #         inst.prob_s = prob_s
            
    #         of_exact, _, _, _ = prb.solve(
    #             dict_data,
    #             #verbose=True
    #         )
    #         res_w[idx_w].append(of_exact)
    #     idx_w += 1



    # COMPARISON:
    # test = Tester()
    # n_scenarios = 1000
    # reward_1 = sam.sample_stoch(
    #     inst,
    #     n_scenarios=n_scenarios
    # )
    # ris1 = test.solve_second_stages(
    #     inst,
    #     sol_exact,
    #     n_scenarios,
    #     reward_1
    # )
    # reward_2 = sam.sample_stoch(
    #     inst,
    #     n_scenarios=n_scenarios
    # )
    # ris2 = test.solve_second_stages(
    #     inst,
    #     sol_exact,
    #     n_scenarios,
    #     reward_2
    # )
    # plot_comparison_hist(
    #     [ris1, ris2],
    #     ["run1", "run2"],
    #     ['red', 'blue'],
    #     "profit", "occurencies"
    # )

    '''
    heu = SimpleHeu(2)
    of_heu, sol_heu, comp_time_heu = heu.solve(
        dict_data
    )
    print(of_heu, sol_heu, comp_time_heu)

    # printing results of a file
    file_output = open(
        "./results/exp_general_table.csv",
        "w"
    )
    file_output.write("method, of, sol, time\n")
    file_output.write("{}, {}, {}, {}\n".format(
        "heu", of_heu, sol_heu, comp_time_heu
    ))
    file_output.write("{}, {}, {}, {}\n".format(
        "exact", of_exact, sol_exact, comp_time_exact
    ))
    file_output.close()
    '''
