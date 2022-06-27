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
        
        w = float(dict_data["w"])
        
        #Initialize time
        start = time.time()
        
        #Mean values over scenarios
        c_ij = np.mean(dict_data["c_sij"], axis=0)#(1 - w)*np.sum(dict_data["prob_s"]*dict_data["c_sij"], axis=0) #- w*np.sqrt(np.sum(dict_data["prob_s"]*np.power(dict_data["c_sij"],2), axis=0))
        p_mj = np.mean(dict_data["p_smj"], axis=0)#(1 - w)*np.sum(dict_data["prob_s"]*dict_data["p_smj"], axis=0) #- w*np.sqrt(np.sum(dict_data["prob_s"]*np.power(dict_data["p_smj"],2), axis=0))
        y_ijk = np.mean(dict_data["y_sijk"], axis=0)#(1 - w)*np.sum(np.matmul(dict_data["prob_s"].reshape(1,4),dict_data["y_sijk"]), axis=0) #- w*np.sqrt(np.sum(dict_data["prob_s"]*np.power(dict_data["y_sijk"],2), axis=0))
        
        #Compute the convenience of each client
        profit_mj = np.multiply(np.array(dict_data["d_mj"]), np.array(dict_data["f_mj"]))
        profit_ordered = np.flip(np.sort(profit_mj, axis=None))
        
        #Harvesting cost for each band and week
        c_ijk = np.zeros((dict_data['crops'],dict_data["weeks"], dict_data["bands"]))
        for k in range(dict_data["bands"]):
            c_ijk[:,:,k] = c_ij
        harv_cost_ijk = np.divide(c_ijk, y_ijk)
        
        #%% FIRST STAGE VARIABLES
        
        #Try to set crops for each client
        for profit in profit_ordered:
            #Get client, week and associated band
            client, week = np.where(profit_mj == profit)
            client = client[0]
            week = week[0]
            band = dict_data["Km"][client]
            demand = dict_data["d_mj"][client][week]
            
            #Get harvesting costs ordered (most convenient band)
            client_harv_cost = harv_cost_ijk[:, week, band]
            client_harv_cost_min = np.amin(client_harv_cost,1) #For each crop the cheapest band to use
            client_harv_cost_band = np.argmin(client_harv_cost, axis=1) #Best bands
            harv_cost_idx = np.argsort(client_harv_cost_min, axis=None)
            
            #Try to satisfy demand on each crop
            needed_dem = demand
            for idx in harv_cost_idx:
                
                #Compute needed area
                y_aux = y_ijk[idx, week, client_harv_cost_band[idx]]
                needed_area = needed_dem/y_aux
                
                #Land limit constraint
                av_area = 10000 - A_i[idx] #Initial available area
                
                #Use available land
                if (av_area >= needed_area):
                    A_i[idx] = A_i[idx] + needed_area
                    needed_area = 0
                else:
                    A_i[idx] = A_i[idx] + av_area
                    needed_area = needed_area - av_area
                
                #Update needed_demand
                needed_dem = y_aux*needed_area
                
                #Break for if the demand has been satisfy
                if (needed_dem <= 0):
                    break
                
        #%% SECOND STAGE VARIABLES
        
        #Auxiliar copy of A_i
        aux_A_i = np.copy(A_i)
        
        #For each scenario
        for s in range(dict_data["scenarios"]):
            
            c_ij = dict_data["c_sij"][s,:,:]
            p_mj = dict_data["p_smj"][s,:,:]
            y_ijk = dict_data["y_sijk"][s,:,:]
            
            #Harvesting cost for each band and week
            c_ijk = np.zeros((dict_data['crops'],dict_data["weeks"], dict_data["bands"]))
            for k in range(dict_data["bands"]):
                c_ijk[:,:,k] = c_ij
            harv_cost_ijk = np.divide(c_ijk, y_ijk)
            
            #Try to harvest for each client
            for profit in profit_ordered:
                #Get client, week and associated band
                client, week = np.where(profit_mj == profit)
                client = client[0]
                week = week[0]
                band = dict_data["Km"][client]
                demand = dict_data["d_mj"][client][week]
                
                #Get harvesting costs ordered (most convenient band)
                client_harv_cost = harv_cost_ijk[:, week, band]
                client_harv_cost_min = np.amin(client_harv_cost,1) #For each crop the cheapest band to use
                client_harv_cost_band = np.argmin(client_harv_cost, axis=1) #Best bands
                harv_cost_idx = np.argsort(client_harv_cost_min, axis=None)
                
                #Try to satisfy demand on each crop
                needed_dem = demand
                for idx in harv_cost_idx:
                    
                    #Compute needed area to harvest from crop {idx}
                    y_aux = y_ijk[idx, week, client_harv_cost_band[idx]]
                    needed_area = needed_dem/y_aux
                    
                    #Harvest
                    harvested_ha = 0
                    if (aux_A_i[idx] >= needed_area):
                        harvested_ha = needed_area
                        aux_A_i[idx] = aux_A_i[idx] - needed_area
                        H_sij[s,idx,week] = H_sij[s,idx,week] + needed_area #H_sij may be used by more than one customer
                        needed_area = 0
                    else:
                        harvested_ha = needed_area - aux_A_i[idx]
                        H_sij[s,idx,week] = H_sij[s,idx,week] + needed_area - aux_A_i[idx]
                        needed_area = needed_area - aux_A_i[idx]
                        aux_A_i[idx] = 0
                    
                    #Sell
                    weight = harvested_ha*y_aux
                    F_sjmk[s, week, client, client_harv_cost_band[idx]] += weight
                    
                    #Update needed_demand
                    needed_dem = needed_dem - weight
                    
                    #Break for if the demand has been satisfy
                    if (needed_dem <= 0):
                        needed_dem = 0
                        break
                
                #Pay unsatisfied customer demand
                P_smj[s, client, week] = needed_dem
            
            #Sell surplus (Marketing contraint)
            for j in range(dict_data["weeks"]):
                for k in range(dict_data["bands"]):
                    S_sjk[s,j,k] = np.sum(F_sjmk[s,j,:,k]) - np.sum(np.multiply(dict_data["y_sijk"][s,:,j,k], H_sij[s,:,j]))
                    
        #Measure execution time
        end = time.time()
        comp_time = end - start
        print("Heuristics time = ", comp_time)
        
        print("OF Heuristics = ", compute_of(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data))
        
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
