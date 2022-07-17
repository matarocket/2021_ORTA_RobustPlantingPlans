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
        term6 += dict_data["c_prime"]*A_i[i]
    
    term7 = 0
    for m in range(dict_data["customers"]):
        for j in range(dict_data["weeks"]):
            term7 += dict_data["p_smj"][s][m][j]*P_smj[s][m][j]
    
    ProfitTerm =  term1 + term2 - term3 + term4 - term5 - term6 - term7
                
    return ProfitTerm

#Ramp function
def ramp_funciton(x):
    if x>0:
        return x
    else:
        return 0

#Compute Expected value of a function
def compute_E_s(function_of_s, dict_data):
    E_s = 0
    for s in range(dict_data["scenarios"]):
        E_s += dict_data["prob_s"][s]*function_of_s(s)
    
    return E_s

#Evaluate solution
def compute_of(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data):
    of = 0
    
    w = dict_data["w"]
    
    E_Profit = 0
    Risk = 0
    
    for s in range(dict_data["scenarios"]):
        E_Profit += dict_data["prob_s"][s]*Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s)
    
    for s in range(dict_data["scenarios"]):
        profit = Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s)
        z = (np.abs(profit - E_Profit))
        Risk += z*dict_data["prob_s"][s]
    
    of = ((1-w)*E_Profit) - w*Risk
    return of

#Load model from Gurobi to manually compute OF and verify constraints
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
    
    #Verify constriants
    #verify_cons(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
    
    #Compute cost function
    of = compute_of(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
    
    return of

#Verify constraints
def verify_cons(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data):
    
    print("Verifying constraints ...")
    
    #Auxiliar flag
    cons_flag = True
    
    #Marketing Constraint
    for s in range(dict_data["scenarios"]):
        for j in range(dict_data["weeks"]):
            for k in range(dict_data["bands"]):
                if(S_sjk[s,j,k] != np.sum(np.multiply(dict_data["y_sijk"][s,:,j,k], H_sij[s,:,j])) - np.sum(F_sjmk[s,j,:,k])):
                    cons_flag = False
                    print("Problems in Marketing constraint!")
                    print("Delta = ", S_sjk[s,j,k] - np.sum(np.multiply(dict_data["y_sijk"][s,:,j,k], H_sij[s,:,j])) - np.sum(F_sjmk[s,j,:,k]))
    
    #Demand constraint
    for s in range(dict_data["scenarios"]):
        for j in range(dict_data["weeks"]):
            for m in range(dict_data["customers"]):
                if(np.sum(F_sjmk[s,j,m,dict_data["Km"][m]]) != P_smj[s,m,j] + dict_data["d_mj"][m,j]):
                    cons_flag = False
                    print("Problems in Demand constraint!")
                    print("Delta = ", np.sum(F_sjmk[s,j,m,dict_data["Km"][m]]) - P_smj[s,m,j] + dict_data["d_mj"][m,j])
    
    #Open market constraint
    for s in range(dict_data["scenarios"]):
        for j in range(dict_data["weeks"]):
            if(np.sum(S_sjk[s,j,:]) > 0.25*np.sum(dict_data["d_mj"][:,j])):
                    cons_flag = False
                    print("Problems in open market constraint!")
                    
    #Land use contraints 1
    if(np.sum(A_i) != dict_data["a"] - L_plus + L_minus):
        cons_flag = False
        print("Problems in land use constraint 1!")
 
    #Land use contraints 2
    for s in range(dict_data["scenarios"]):
        for i in range(dict_data["crops"]):
            if(A_i[i] != np.sum(H_sij[s,i,:])):
                cons_flag = False
                print("Problems in land use constraint 2!")
    
    #Disease constraint
    for j in range(dict_data["scenarios"]):
        for q in range(dict_data["weeks"]):
            for s in range(dict_data["customers"]):
                pass
    
    #Individual Variety Limit
    for v in range(dict_data["varieties"]):
        aux=[]
        for i in range(dict_data["crops"]):
             if (dict_data["Ai_dict"][i]["Variety"] == v):
                 aux.append(i)
        if(np.sum(A_i[aux]) > 0.4*np.sum(A_i)):
            cons_flag = False
            print("Problems in Individual Variety Limit!")
    
    #Individual crop limit
    for i in range(dict_data["crops"]):
        if(A_i[i] > 0.2*np.sum(A_i)):
            cons_flag = False
            print("Problems in Individual crop limit!")
        
    return cons_flag
    
                

#Function for computing expected value of an s-dependent matrix
def collapse_prob(mat, prob_s):
    e_s = np.copy(mat)
    for s in range(len(prob_s)):
        e_s[s] = prob_s[s]*e_s[s]
    e_s = np.sum(e_s, axis=0)
    return e_s

class SimpleHeu_old():
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
        
        #w from data
        w = float(dict_data["w"])
        
        #Compute "cost" of scenario-dependent variables
        c_ij = (1 - w)*collapse_prob(dict_data["c_sij"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["c_sij"] - collapse_prob(dict_data["c_sij"], dict_data["prob_s"])), dict_data["prob_s"])
        p_mj = (1 - w)*collapse_prob(dict_data["p_smj"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["p_smj"] - collapse_prob(dict_data["p_smj"], dict_data["prob_s"])), dict_data["prob_s"])
        y_ijk = (1 - w)*collapse_prob(dict_data["y_sijk"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["y_sijk"] - collapse_prob(dict_data["y_sijk"], dict_data["prob_s"])), dict_data["prob_s"])
        
        #Compute the profit from each client
        profit_mj = np.multiply(np.array(dict_data["d_mj"]), np.array(dict_data["f_mj"]))
        profit_ordered = np.flip(np.sort(profit_mj, axis=None))
        
        #Harvesting cost for each band and week
        c_ijk = np.zeros((dict_data['crops'],dict_data["weeks"], dict_data["bands"]))
        for k in range(dict_data["bands"]):
            c_ijk[:,:,k] = c_ij
        harv_cost_ijk = np.divide(c_ijk, y_ijk)
        
        #Initialize time
        start = time.time()
        aux_time = start
        
        #%% FIRST STAGE VARIABLES
        
        #Rent the max land possible
        L_minus = dict_data["a"]/10
        
        #Try to set crops for each client
        for profit in profit_ordered:
            #Get client, week and associated band
            client, week = np.where(profit_mj == profit)
            client = client[0]
            week = week[0]
            band = dict_data["Km"][client] #Array of prefered bands
            demand = dict_data["d_mj"][client][week]
            
            #Get harvesting costs ordered (most convenient band)
            client_harv_cost = harv_cost_ijk[:, week, band] #Week is an int, band is an array
            client_harv_cost_min = np.amin(client_harv_cost,1) #For each crop the cheapest band to use
            client_harv_cost_band = np.argmin(client_harv_cost, axis=1) #Best bands
            harv_cost_idx = np.argsort(client_harv_cost_min, axis=None)
            #harv_cost_idx = list(range(len(client_harv_cost_min)))
            
            #Try to satisfy demand on each crop
            needed_dem = demand
            for idx in harv_cost_idx:
                
                #Compute needed area
                y_aux = y_ijk[idx, week, client_harv_cost_band[idx]]
                needed_area = needed_dem/y_aux
                
                #Land limit constraint
                av_area = dict_data["a"] + L_minus - np.sum(A_i) #- A_i[idx] #Initial available area
                
                #Individual variety limit
                aux=[]
                for i in range(dict_data["crops"]):
                     if (dict_data["Ai_dict"][i]["Variety"] == dict_data["Ai_dict"][idx]["Variety"]):
                         aux.append(i)
                v_land = np.sum(A_i[aux])
                av_area = np.min([av_area, 0.4*dict_data["a"] - v_land])
                
                #Individual crop limit
                av_area = np.min ([av_area, 0.2*dict_data["a"]])
                
                #Final area to seed and stisfiable demand in tons
                area_to_seed = np.min([av_area, needed_area])
                demand_from_area = y_aux*area_to_seed
                
                #Check if sowing brings a benefit
                #if((dict_data["c_prime"]*area_to_seed) < (p_mj[client, week]*y_aux*area_to_seed)):
                if (dict_data["f_mj"][client,week]*demand_from_area - dict_data["c_prime"]*area_to_seed - p_mj[client, week]*demand_from_area > 0):
                #if True:
                    
                    #Use available land
                    seeded_ha = area_to_seed
                    A_i[idx] = A_i[idx] + seeded_ha
                    needed_area = needed_area - seeded_ha
                
                #Update needed_demand
                needed_dem = y_aux*needed_area
                
                #Break for if the demand has been satisfy
                if (needed_dem <= 0):
                    break
            
        #Try to set crops to sell surplus
        # --> This is not done to avoid not respecting the open market constraint
        
        #Compute rentable land
        #L_plus = dict_data["a"] - np.sum(A_i)
        print("First stage time = ", time.time() - aux_time)
        aux_time = time.time()
                
        #%% SECOND STAGE VARIABLES
        
        #For each scenario
        for s in range(dict_data["scenarios"]):
            
            #Auxiliar copy of A_i
            aux_A_i = np.copy(A_i)
            #aux_A_i = A_i
            
            #Pick costs for the actual scenario
            c_ij = dict_data["c_sij"][s,:,:]
            p_mj = dict_data["p_smj"][s,:,:]
            y_ijk = dict_data["y_sijk"][s,:,:,:]
            
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
                aux_time = time.time()
                #Get harvesting costs ordered (most convenient band)
                client_harv_cost = harv_cost_ijk[:, week, band]
                client_harv_cost_min = np.amin(client_harv_cost,1) #For each crop the cheapest band to use
                client_harv_cost_band = np.argmin(client_harv_cost, axis=1) #Best bands
                #harv_cost_idx = np.argsort(client_harv_cost_min, axis=None)
                harv_cost_idx = list(range(len(client_harv_cost_min)))
                print("Harvesting cost times = ", time.time() - aux_time, " (harvest) - s =", s)
                
                #Try to satisfy demand on each crop
                needed_dem = demand
                for idx in harv_cost_idx:
                    
                    #Compute needed area to harvest from crop {idx}
                    y_aux = y_ijk[idx, week, client_harv_cost_band[idx]]
                    needed_area = needed_dem/y_aux
                    
                    #Sold weight
                    weight = 0
                    
                    # #If harvesting this crop is benefitial, proceed to sow
                    # if((dict_data["f_mj"][client, week]*y_aux*needed_area - dict_data["c_prime"]*needed_area) > (p_mj[client, week]*y_aux*needed_area) and
                    #     (profit > needed_area*c_ijk[idx, week, client_harv_cost_band[idx]])):
                    
                    if (dict_data["f_mj"][client,week]*needed_dem - dict_data["c_prime"]*needed_area - p_mj[client, week]*needed_dem > 0):
                        #Harvest
                        harvested_ha = np.min([aux_A_i[idx], needed_area])
                        aux_A_i[idx] = aux_A_i[idx] - harvested_ha
                        H_sij[s,idx,week] = H_sij[s,idx,week] + harvested_ha
                        needed_area = needed_area - harvested_ha
                        
                        #Sell
                        weight = harvested_ha*y_aux
                        F_sjmk[s, week, client, client_harv_cost_band[idx]] += weight
                    
                    #Update needed_demand
                    needed_dem = needed_dem - weight
                    
                    #Break for if the demand has been satisfy
                    if (needed_dem <= 0):
                        needed_dem = 0
                        
                #Pay unsatisfied customer demand
                if (np.sum(F_sjmk[s, week, client, dict_data["Km"][client]]) < dict_data["d_mj"][client,week]):
                    weigth_to_buy = dict_data["d_mj"][client,week] - np.sum(F_sjmk[s, week, client, dict_data["Km"][client]])
                    F_sjmk[s, week, client, dict_data["Km"][client][0]] += weigth_to_buy
                    P_smj[s, client, week] += weigth_to_buy
                    
            
            #Sell surplus (Marketing contraint)
            for j in range(dict_data["weeks"]):
                for k in range(dict_data["bands"]):
                    S_sjk[s,j,k] = np.sum(np.multiply(dict_data["y_sijk"][s,:,j,k], H_sij[s,:,j])) - np.sum(F_sjmk[s,j,:,k])
                    pass
            
            print("Second stage time = ", time.time() - aux_time, " - s =", s)
            aux_time = time.time()
        
                
        #Measure execution time
        end = time.time()
        comp_time = end - start
        print("Heuristics time = ", comp_time)
        
        #Compute cost function
        of = compute_of(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
        print("OF Heuristics = ", of)
        
        profit = 0
        for s in range(dict_data["scenarios"]):
            profit += dict_data["prob_s"][s]*Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s)
        print("Expected profit = ", profit)
        
        #verify_cons(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
        
        #Load solution
        sol_x = A_i
        
        # sol_x = [0] * dict_data['n_items']
        # of = -1
        
        # start = time.time()
        # ratio = [0] * dict_data['n_items']
        # for i in range(dict_data['n_items']):
        #     ratio[i] = dict_data['profits'][i] / dict_data['sizes'][i]
        # sorted_pos = [ratio.index(x) for x in sorted(ratio)]
        # sorted_pos.reverse()
        # cap_tmp = 0
        # for i, item in enumerate(sorted_pos):
        #     cap_tmp += dict_data['sizes'][item]
        #     if cap_tmp > dict_data['max_size']:
        #         break
        #     sol_x[item] = 1
        # end = time.time()
        
        return of, sol_x, comp_time

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
        
        #w from data
        w = float(dict_data["w"])
        
        #Compute "cost" of scenario-dependent variables
        c_ij = (1 - w)*collapse_prob(dict_data["c_sij"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["c_sij"] - collapse_prob(dict_data["c_sij"], dict_data["prob_s"])), dict_data["prob_s"])
        p_mj = (1 - w)*collapse_prob(dict_data["p_smj"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["p_smj"] - collapse_prob(dict_data["p_smj"], dict_data["prob_s"])), dict_data["prob_s"])
        y_ijk = (1 - w)*collapse_prob(dict_data["y_sijk"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["y_sijk"] - collapse_prob(dict_data["y_sijk"], dict_data["prob_s"])), dict_data["prob_s"])
        
        #Compute the profit from each client
        profit_mj = np.multiply(np.array(dict_data["d_mj"]), np.array(dict_data["f_mj"]))
        profit_ordered = np.flip(np.sort(profit_mj, axis=None))
        
        #Harvesting cost for each band and week
        c_ijk = np.zeros((dict_data['crops'],dict_data["weeks"], dict_data["bands"]))
        for k in range(dict_data["bands"]):
            c_ijk[:,:,k] = c_ij
        harv_cost_ijk = np.divide(c_ijk, y_ijk)
        
        #Initialize time
        start = time.time()
        aux_time = start
        
        #%% FIRST STAGE VARIABLES
        
        #Rent the max land possible
        L_minus = dict_data["a"]/10
        
        #Try to set crops for each client
        for profit in profit_ordered:
            #Get client, week and associated band
            client, week = np.where(profit_mj == profit)
            client = client[0]
            week = week[0]
            band = dict_data["Km"][client] #Array of prefered bands
            demand = dict_data["d_mj"][client][week]
            
            # #Get harvesting costs ordered (most convenient band)
            # client_harv_cost = harv_cost_ijk[:, week, band] #Week is an int, band is an array
            # client_harv_cost_min = np.amin(client_harv_cost,1) #For each crop the cheapest band to use
            # client_harv_cost_band = np.argmin(client_harv_cost, axis=1) #Best bands
            # #harv_cost_idx = np.argsort(client_harv_cost_min, axis=None)
            # harv_cost_idx = list(range(len(client_harv_cost_min)))
            
            #Get the crop that produces the most
            crops_production = y_ijk[:, week, band] #Week is an int, band is an array
            crops_production_max = np.amax(crops_production,0) #For each crop the cheapest band to use
            crops_production_idx = np.argmax(crops_production, axis=0) #Best bands
            best_band_idx = np.argmax(crops_production_max, axis=0)
            crops_production_idx = crops_production_idx[best_band_idx]
            best_band = band[best_band_idx]
                
            needed_dem = demand 
            
            #Compute needed area
            y_aux = y_ijk[crops_production_idx, week, best_band]
            needed_area = needed_dem/y_aux
            
            #Land limit constraint
            av_area = dict_data["a"] + L_minus - np.sum(A_i) #- A_i[idx] #Initial available area
            
            #Individual variety limit
            aux=[]
            for i in range(dict_data["crops"]):
                 if (dict_data["Ai_dict"][i]["Variety"] == dict_data["Ai_dict"][crops_production_idx]["Variety"]):
                     aux.append(i)
            v_land = np.sum(A_i[aux])
            av_area = np.min([av_area, 0.4*dict_data["a"] - v_land])
            
            #Individual crop limit
            av_area = np.min ([av_area, 0.2*dict_data["a"]])
            
            #Final area to seed and stisfiable demand in tons
            area_to_seed = np.min([av_area, needed_area])
            demand_from_area = y_aux*area_to_seed
            
            #Check if sowing brings a benefit
            if((dict_data["c_prime"]*area_to_seed) < (p_mj[client, week]*demand_from_area)):
            #if (dict_data["f_mj"][client,week]*demand_from_area - dict_data["c_prime"]*area_to_seed - p_mj[client, week]*demand_from_area > 0):
            #if True:
                
                #Use available land
                seeded_ha = area_to_seed
                A_i[crops_production_idx] = A_i[crops_production_idx] + seeded_ha
                needed_area = needed_area - seeded_ha
            
            #Update needed_demand
            needed_dem = y_aux*needed_area
            
        #Try to set crops to sell surplus
        # --> This is not done to avoid not respecting the open market constraint
        
        #Compute rentable land
        #L_plus = dict_data["a"] - np.sum(A_i)
        print("First stage time = ", time.time() - aux_time)
        aux_time = time.time()
                
        #%% SECOND STAGE VARIABLES
        
        #For each scenario
        for s in range(dict_data["scenarios"]):
            
            #Auxiliar copy of A_i
            aux_A_i = np.copy(A_i)
            #aux_A_i = A_i
            
            #Pick costs for the actual scenario
            c_ij = dict_data["c_sij"][s,:,:]
            p_mj = dict_data["p_smj"][s,:,:]
            y_ijk = dict_data["y_sijk"][s,:,:,:]
            
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
                aux_time = time.time()
                
                # #Get harvesting costs ordered (most convenient band)
                # client_harv_cost = harv_cost_ijk[:, week, band]
                # client_harv_cost_min = np.amin(client_harv_cost,1) #For each crop the cheapest band to use
                # client_harv_cost_band = np.argmin(client_harv_cost, axis=1) #Best bands
                
                # #harv_cost_idx = np.argsort(client_harv_cost_min, axis=None)
                # harv_cost_idx = list(range(len(client_harv_cost_min)))
                # print("Harvesting cost times = ", time.time() - aux_time, " (harvest) - s =", s)
                
                #Get the crop that produces the most
                crops_production = y_ijk[:, week, band] #Week is an int, band is an array
                crops_production_max = np.amax(crops_production,0) #For each crop the cheapest band to use
                crops_production_idx = np.argmax(crops_production, axis=0) #Best bands
                best_band_idx = np.argmax(crops_production_max, axis=0)
                crops_production_idx = crops_production_idx[best_band_idx]
                best_band = band[best_band_idx]
                
                #Try to satisfy demand on each crop
                needed_dem = demand
                    
                #Compute needed area to harvest from crop {idx}
                y_aux = y_ijk[crops_production_idx, week, best_band]
                needed_area = needed_dem/y_aux
                
                #Sold weight
                weight = 0
                
                # #If harvesting this crop is benefitial, proceed to sow
                # if((dict_data["f_mj"][client, week]*y_aux*needed_area - dict_data["c_prime"]*needed_area) > (p_mj[client, week]*y_aux*needed_area) and
                #     (profit > needed_area*c_ijk[idx, week, client_harv_cost_band[idx]])):
                
                if (dict_data["f_mj"][client,week]*needed_dem - dict_data["c_prime"]*needed_area - p_mj[client, week]*needed_dem > 0):
                    #Harvest
                    harvested_ha = np.min([aux_A_i[crops_production_idx], needed_area])
                    aux_A_i[crops_production_idx] = aux_A_i[crops_production_idx] - harvested_ha
                    H_sij[s,crops_production_idx,week] = H_sij[s,crops_production_idx,week] + harvested_ha
                    needed_area = needed_area - harvested_ha
                    
                    #Sell
                    weight = harvested_ha*y_aux
                    F_sjmk[s, week, client, best_band] += weight
                
                #Update needed_demand
                needed_dem = needed_dem - weight
                
                #Break for if the demand has been satisfy
                if (needed_dem <= 0):
                    needed_dem = 0
                    
            #Pay unsatisfied customer demand
            if (np.sum(F_sjmk[s, week, client, dict_data["Km"][client]]) < dict_data["d_mj"][client,week]):
                weigth_to_buy = dict_data["d_mj"][client,week] - np.sum(F_sjmk[s, week, client, dict_data["Km"][client]])
                F_sjmk[s, week, client, dict_data["Km"][client][0]] += weigth_to_buy
                P_smj[s, client, week] += weigth_to_buy
                    
            
            #Sell surplus (Marketing contraint)
            for j in range(dict_data["weeks"]):
                for k in range(dict_data["bands"]):
                    S_sjk[s,j,k] = np.sum(np.multiply(dict_data["y_sijk"][s,:,j,k], H_sij[s,:,j])) - np.sum(F_sjmk[s,j,:,k])
                    pass
            
            print("Second stage time = ", time.time() - aux_time, " - s =", s)
            aux_time = time.time()
        
                
        #Measure execution time
        end = time.time()
        comp_time = end - start
        print("Heuristics time = ", comp_time)
        
        #Compute cost function
        of = compute_of(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
        print("OF Heuristics = ", of)
        
        profit = 0
        for s in range(dict_data["scenarios"]):
            profit += dict_data["prob_s"][s]*Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s)
        print("Expected profit = ", profit)
        
        #verify_cons(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
        
        #Load solution
        sol_x = A_i
        
        # sol_x = [0] * dict_data['n_items']
        # of = -1
        
        # start = time.time()
        # ratio = [0] * dict_data['n_items']
        # for i in range(dict_data['n_items']):
        #     ratio[i] = dict_data['profits'][i] / dict_data['sizes'][i]
        # sorted_pos = [ratio.index(x) for x in sorted(ratio)]
        # sorted_pos.reverse()
        # cap_tmp = 0
        # for i, item in enumerate(sorted_pos):
        #     cap_tmp += dict_data['sizes'][item]
        #     if cap_tmp > dict_data['max_size']:
        #         break
        #     sol_x[item] = 1
        # end = time.time()
        
        return of, sol_x, comp_time
