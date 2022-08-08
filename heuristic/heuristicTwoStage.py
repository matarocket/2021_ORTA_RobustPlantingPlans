# -*- coding: utf-8 -*- 
import time
import math
import logging
import numpy as np

def allocateSowing(dict_data, sowing,L_minus,A_i, client, week, band,demand):
    
    #Get the crop that produces the most
    crops_production = sowing[:, week, band] #Week is an int, band is an array
    crops_production_max = np.amax(crops_production,0) #For each crop use the most productive one
    crops_production_idx = np.argmax(crops_production, axis=0) #Best bands
    best_band_idx = np.argmax(crops_production_max, axis=0)
    crops_production_idx = crops_production_idx[best_band_idx]
    best_band = band[best_band_idx]
        
    needed_dem = demand 
    
    #Compute needed area
    y_aux = sowing[crops_production_idx, week, best_band]
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
    av_area = np.min([av_area, 0.2*dict_data["a"]])

    #Remove taken crops
    sowing[crops_production_idx, :, :] = -1
    return av_area,crops_production_idx,needed_area,best_band,sowing

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
        assigned_crop = {}
        planting_cost_total = 0

        #w from data
        w = float(dict_data["w"])
        
        #Compute "cost" of scenario-dependent variables
        #c_ij = (1 - w)*collapse_prob(dict_data["c_sij"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["c_sij"] - collapse_prob(dict_data["c_sij"], dict_data["prob_s"])), dict_data["prob_s"])
        #p_mj = (1 - w)*collapse_prob(dict_data["p_smj"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["p_smj"] - collapse_prob(dict_data["p_smj"], dict_data["prob_s"])), dict_data["prob_s"])
        #y_ijk = (1 - w)*collapse_prob(dict_data["y_sijk"], dict_data["prob_s"]) - w*collapse_prob(np.abs(dict_data["y_sijk"] - collapse_prob(dict_data["y_sijk"], dict_data["prob_s"])), dict_data["prob_s"])
        c_prime = dict_data["c_prime"]
        c_sij = dict_data["c_sij"]
        y_ijk = dict_data["y_sijk"][0,:,:,:]
        sowing = (dict_data["y_sijk"][0,:,:,:]).copy()
        #Compute the profit from each client
        profit_mj = np.multiply(np.array(dict_data["d_mj"]), np.array(dict_data["f_mj"]))
        profit_ordered = np.flip(np.sort(profit_mj, axis=None))
        
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
            
            
            av_area,crops_production_idx,needed_area,best_band,sowing = allocateSowing(dict_data, sowing,L_minus,A_i, client, week, band,demand)

            #Try to set crops to sell surplus respecting the open market constraint (25% of the client demand for week j)
            area_to_seed = min(av_area,needed_area*1,25)
            #demand_from_area = y_aux*area_to_seed
            
            #Use available land
            seeded_ha = area_to_seed
            planting_cost_total += c_prime * seeded_ha
            A_i[crops_production_idx] = A_i[crops_production_idx] + seeded_ha
            needed_area = needed_area - seeded_ha
            assigned_crop[crops_production_idx] = {"client":client,"demand":demand,"week":week,"band":best_band}
            #Update needed_demand
            #needed_dem = y_aux*needed_area
            
        
        #Compute rentable land
        #L_plus = dict_data["a"] - np.sum(A_i)
        print("First stage time = ", time.time() - aux_time)
        aux_time = time.time()
                
        #%% SECOND STAGE VARIABLES
        
        #For each scenario
        profitEuristic = 0
        total_harv_cost = 0
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
            
            aux_time = time.time()
            #Try to harvest for each client
            for crop_idx in assigned_crop:
                demand = assigned_crop[crop_idx]["demand"]
                client = assigned_crop[crop_idx]["client"]
                band = assigned_crop[crop_idx]["band"]
                week = assigned_crop[crop_idx]["week"]

                crop_yield_demanded = demand
                crop_yield_real = y_ijk[crop_idx,week,band] * A_i[crop_idx]

                #Calculate profit
                profitEuristic += crop_yield_real * dict_data["f_mj"][client, week]
                #Evaluate the difference between demanded yield and actual yield    
                yield_difference = crop_yield_real - crop_yield_demanded
                #Based on yield difference, buy or sell on the open market to satisfy client request or to sell surplus
                profitEuristic += dict_data["s_sj"][s,week] * yield_difference
                #Calculate the harvesting cost given the actual scenario 
                total_harv_cost += c_sij[s, crop_idx, week] * A_i[crop_idx]
           
        print("Second stage time = ", time.time() - aux_time, " - s =", s)
        aux_time = time.time()
                 
        #Measure execution time
        end = time.time()
        comp_time = end - start
        print("Heuristics time = ", comp_time)
        
        #Compute cost function
        #of = compute_of(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data)
        #print("OF Heuristics = ", of)
        
        '''profit = 0
        for s in range(dict_data["scenarios"]):
            profit += dict_data["prob_s"][s]*Profit_s(F_sjmk, H_sij, S_sjk, L_minus, L_plus, P_smj, A_i, dict_data, s)
        print("Expected profit = ", profit)'''
        
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
        of=1 #DEBUG 
        print("Planting cost ", planting_cost_total)
        print("Harvesting cost ",total_harv_cost/4)
        print("The profit is: ",(profitEuristic-total_harv_cost)/(s+1)-planting_cost_total)
        return of, sol_x, comp_time
