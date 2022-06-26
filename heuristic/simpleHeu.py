# -*- coding: utf-8 -*-
import time
import math
import logging
import numpy as np


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
        term6 = dict_data["c_prime"]*A_i[i]
    
    term7 = 0
    for m in range(dict_data["customers"]):
        for j in range(dict_data["weeks"]):
            term7 += dict_data["p_smj"][s][m][j]*P_smj[s][m][j]
    
    ProfitTerm =  term1 + term2 - term3 + term4 - term5 - term6 - term7
                
    return ProfitTerm

#Compute Expected value of a function
def compute_E_s(function_of_s, dict_data):
    E_s = 0
    for s in range(dict_data["scenarios"]):
        E_s += dict_data["prob_s"][s]*function_of_s(s)
    
    return E_s

#Evaluate solution
def compute_of(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data):
    of = 0
    
    w = 0.5
    
    
    E_Profit = 0
    Risk = 0
    
    for s in range(dict_data["scenarios"]):
        E_Profit += dict_data["prob_s"][s]*Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s)
    
    z=[]
    for s in range(dict_data["scenarios"]):
        profit = Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s)
        z.append(np.abs(profit - E_Profit))
    for s in range(dict_data["scenarios"]):
        Risk += z[s]*dict_data["prob_s"][s]
    
    of = ((1-w)*E_Profit) - w*Risk
    return of

def Load_sol_from_gb(model, dict_data):
    
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
    
    of = compute_of(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
    
    return of
    

class SimpleHeu():
    def __init__(self):
        pass

    def solve(
        self, dict_data, reward, n_scenarios,
    ):
        
        #Initialization of control variables
        F_sjmk = np.zeros((dict_data["scenarios"],dict_data["weeks"],dict_data["customers"],dict_data["bands"]))
        H_sij = np.zeros((dict_data["scenarios"],dict_data['crops'],dict_data["weeks"]))
        S_sjk = np.zeros((dict_data["scenarios"],dict_data["weeks"],dict_data["bands"]))
        L_minus = 0
        L_plus = 0
        P_smj = np.zeros((dict_data["scenarios"],dict_data["customers"],dict_data["weeks"]))
        A_i = np.zeros((dict_data['crops']))
        
        #Initialize time
        start = time.time()
        
        #Pick most probable scenario
        s_max = np.argmax(dict_data["prob_s"])
        
        #Maximize profit form clients demand
        profit =  np.multiply(np.array(dict_data["d_mj"]), np.array(dict_data["f_mj"]))
        
        sol_x = [0] * dict_data['n_items']
        of = -1
        
        start = time.time()
        ratio = [0] * dict_data['n_items']
        for i in range(dict_data['n_items']):
            ratio[i] = dict_data['profits'][i] / dict_data['sizes'][i]
        sorted_pos = [ratio.index(x) for x in sorted(ratio)]
        sorted_pos.reverse()
        cap_tmp = 0
        for i, item in enumerate(sorted_pos):
            cap_tmp += dict_data['sizes'][item]
            if cap_tmp > dict_data['max_size']:
                break
            sol_x[item] = 1
        end = time.time()

        comp_time = end - start
        
        return of, sol_x, comp_time
