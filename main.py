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
from heuristic.firstStageHeuristicALNS import Heuristic
from solver.sampler import Sampler
from utility.plot_results import plot_comparison_hist
from utility.plot_results import plot_w_comparison
import matplotlib.pyplot as plt

np.random.seed(0)

#Profit for a given scenario
def Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s):
    
    term1 = 0
    for k in range(dict_data['bands']):
        for j in range(dict_data['weeks']):
            term1 += dict_data["s_sj"][s][j] * S_sjk[s][j][k]
    
    term2 = 0
    for m in range(dict_data["customers"]):
        for j in range(dict_data["weeks"]):
            for k in range(dict_data["bands"]):
                term2 += dict_data["f_mj"][m][j] * F_sjmk[s][j][m][k]
    
    term3 = 0
    for i in range(dict_data["crops"]):
        for j in range(dict_data["weeks"]):
            term3 += dict_data["c_sij"][s][i][j]*H_sij[s][i][j]
    
    term4 = dict_data["c_plus"]*L_plus
    term5 = dict_data["c_minus"]*L_minus
    
    term6 = 0
    for i in range(dict_data["crops"]):
        term6 += dict_data["c_prime"]*A_i[i]
    
    term7 = 0
    for m in range(dict_data["customers"]):
        for j in range(dict_data["weeks"]):
            term7 += dict_data["p_smj"][s][m][j]*P_smj[s][m][j]
    
    ProfitTerm =  term1 + term2 - term3 + term4 - term5 - term6 - term7
                
    return ProfitTerm

#Compute Expected value of a function
def compute_mean_profit(prob_s, F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data):
    E_s = 0
    for s in range(dict_data["scenarios"]):
        E_s += prob_s[s]*Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s)
    
    return E_s

def Mean_profit_from_gb(model, dict_data, prob_s):
    
    scenarios = range(dict_data['scenarios'])
    weeks = range(dict_data['weeks'])
    bands = range(dict_data['bands'])
    customers = range(dict_data["customers"]) 
    crops = range(dict_data["crops"]) 
        
    F_sjmk = np.zeros((dict_data["scenarios"],dict_data["weeks"],dict_data["customers"],dict_data["bands"]))
    H_sij = np.zeros((dict_data["scenarios"],dict_data['crops'],dict_data["weeks"]))
    S_sjk = np.zeros((dict_data["scenarios"],dict_data["weeks"],dict_data["bands"]))
    L_minus = 0
    L_plus = 0
    P_smj = np.zeros((dict_data["scenarios"],dict_data["customers"],dict_data["weeks"]))
    A_i = np.zeros((dict_data['crops']))
    
    for s in scenarios:
        for j in weeks:
            for m in customers:
                for k in bands:
                    grb_var = model.getVarByName(
                        f"Fsjmk[{s},{j},{m},{k}]"
                    )
                    F_sjmk[s][j][m][k] = grb_var.X
    
    for s in scenarios:
        for i in crops:
            for j in weeks:
                grb_var = model.getVarByName(
                    f"Hsij[{s},{i},{j}]"
                )
                H_sij[s][i][j] = grb_var.X
    
    for s in scenarios:
        for j in weeks:
            for k in bands:
                grb_var = model.getVarByName(
                    f"Ssjk[{s},{j},{k}]"
                )
                S_sjk[s][j][k] = grb_var.X
    
    grb_var = model.getVarByName("Lminus[0]") 
    L_minus = grb_var.X
    grb_var = model.getVarByName("Lplus[0]") 
    L_plus = grb_var.X
    
    for s in scenarios:
        for m in customers:
            for j in weeks:
                grb_var = model.getVarByName(
                    f"Psmj[{s},{m},{j}]"
                )
                P_smj[s][m][j] = grb_var.X
    
    for i in crops:
        grb_var = model.getVarByName(
            f"Ai[{i}]"
        )
        A_i[i] = grb_var.X
    
    #Verify constriants
    #verify_cons(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
    
    #Compute cost function
    ret = compute_mean_profit(prob_s, F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
    
    return ret

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
    # of_exact, sol_exact, comp_time_exact, model = prb.solve(
    #     dict_data,
    #     prob_s=prob_s
    #     #verbose=True
    # )
    
    #%% IN SAMPLE STABILITY
    
    test = Tester()
    n_scenarios_vector = [12, 15]
    in_sample_res = []
    for n_scenarios in n_scenarios_vector:
        n_repetitions = 50
        of_mean, of_std = test.in_sample_stability(sim_setting, prb, sam, inst, dict_data, n_repetitions, n_scenarios)
        in_sample_res.append((of_mean, of_std))

    # # OUT OF SAMPLE STABILITY 
    # n_scenarios_vector = [5,10,20]
    # out_sample_res = []
    # n_scenarios = 5
    # for n_scenarios in n_scenarios_vector:
    #     n_scenarios_out = 25
    #     of_mean, of_std = test.out_of_sample_stability(sim_setting, prb, sam, inst, dict_data, n_repetitions, n_scenarios, n_scenarios_out)
    #     out_sample_res.append((of_mean, of_std))

    # of_heu, sol_heu, comp_time_second, comp_time_first = Heuristic.solve(dict_data, prob_s)

    # print("\n\n\n>> Profit exact solver :  ", of_exact," <<\n")
    # print(">> Sowing plan (A_i) :  ", sol_exact," <<\n")
    # print(">> Total computational time :  ", comp_time_exact," <<\n")

    # print("\n\n\n>> Profit heuristic solver :  ", of_heu," <<\n")
    # print(">> Heuristic sowing plan (A_i) :  ", sol_heu," <<\n")
    # print(">> First stage computational time for heuristic :  ", comp_time_first," <<")
    # print(">> Second stage computational time for heuristic :  ", comp_time_second," <<")
    # print(">> Total computational time for heuristic :  ", comp_time_second+comp_time_first," <<\n\n\n")

    # print(">> Percentual difference :  ", 100*(of_exact - of_heu)/of_exact, " <<")
    
    
    #%% VARYING W
    
    # w_vector = np.linspace(0,0.9,10)
    # N = 25
    # res_w_exact = np.zeros((N, len(w_vector)))
    # res_w_heu = np.zeros((N, len(w_vector)))
    
    # for n in range(N): 
    #     print(" ------- Starting problem n = ", n, " -------")  
            
    #     #Instance generation
    #     inst = Instance(sim_setting)
        
    #     #Scenario probability
    #     prob_s = sam.sample_stoch(inst)
        
    #     for idx_w, w in enumerate(w_vector):
            
    #         #Set w    
    #         inst.w = w
    #         dict_data = inst.get_data()
            
    #         #Exact solution
    #         of_exact, _, _, opt_model = prb.solve(
    #             dict_data,
    #             prob_s,
    #             #verbose=True
    #         )
            
    #         profit = Mean_profit_from_gb(opt_model, dict_data, prob_s)
            
    #         #Load results matrix
    #         res_w_exact[n, idx_w] = profit#of_exact
    #         #res_w_heu[n, idx_w] = of_heu
           
    # #Exact results
    # w_res_norm = (res_w_exact)*(1./res_w_exact[:,0].reshape(-1,1))
    # mean_w_exact = np.mean(w_res_norm, axis=0)
    # stddev_w_exact = np.std(res_w_exact, axis=0)
    # stddev_w_norm = stddev_w_exact[0]*(1./stddev_w_exact)
    
    # plot_w_comparison(w_vector, mean_w_exact, stddev_w_norm)
    
    
    #%% COMPARISON:
    
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
