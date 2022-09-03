# -*- coding: utf-8 -*- 
import time
import math
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB




class SecondStageSolver():
    def __init__(self):
        pass

    #Function to solve the second single-scenario stage through Gurobi
    def solve(
            self, dict_data,scenario, ai, l_plus, l_minus, verbose=False
            ):
        
        #Load data from configuration file
        weeks = range(dict_data['weeks'])
        bands = range(dict_data['bands'])
        customers = range(dict_data["customers"]) 
        crops = range(dict_data["crops"]) 
        diseases = range(dict_data["diseases"]) 
        varieties = range(dict_data["varieties"]) 
        w = dict_data["w"]

        #Problem name logging
        problem_name = "SecondStageSolverRobustPlantingPlan"
        logging.info("{}".format(problem_name))

        #Creation of model
        model = gp.Model(problem_name)

        #%% Control variables definition
        
        #Weight of sprouts sold by {scenario, week, customer, band}
        Fjmk = model.addVars(
                dict_data["weeks"],dict_data["customers"],dict_data["bands"],
                vtype=GRB.CONTINUOUS,
                name='Fjmk'
            )
        
        #Area of harvested crop by {scenario, crop, week}
        Hij = model.addVars(
                dict_data['crops'],dict_data["weeks"],
                lb=0,
                vtype=GRB.CONTINUOUS,
                name='Hij'
            )
        
        #Weight of surplus-to-demand sprouts by {scemario, week, band}
        Sjk = model.addVars(
                dict_data["weeks"],dict_data["bands"],
                lb=0,
                vtype=GRB.CONTINUOUS,
                name='Sjk'
            ) 
        
    
        #Shortage in demand by {scenario, customer, week}
        Pmj = model.addVars(
                dict_data["customers"],dict_data["weeks"],
                vtype=GRB.CONTINUOUS,
                name='Pmj'
            )

        #overproduction
        Ojk = model.addVars(
            dict_data["weeks"],dict_data["bands"],
            vtype=GRB.CONTINUOUS,
            name='Ojk'
        )

        #Unusable harvesting
        OHi= model.addVars(
            dict_data["crops"],
            vtype=GRB.CONTINUOUS,
            name='OHi'
        )
        
        
        #%% Definition of the cost function
        
        #Profit for a given scenario
        def Profit():
            term1 = (gp.quicksum( gp.quicksum(dict_data["s_sj"][scenario][j] * Sjk[j,k] for j in weeks )for k in bands))
            term2 = gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fjmk[j,m,k] for m in customers)for j in weeks )for k in bands)
            term3 = gp.quicksum( gp.quicksum(dict_data["c_sij"][scenario][i][j]*Hij[i,j] for i in crops )for j in weeks)
            term4 = dict_data["c_plus"]*l_plus
            term5 = dict_data["c_minus"]*l_minus
            term6 = gp.quicksum(dict_data["c_prime"]*ai[i] for i in crops)
            term7 = gp.quicksum( gp.quicksum(dict_data["p_smj"][scenario][m][j]*Pmj[m,j] for m in customers )for j in weeks)
            ProfitTerm =  term1 + term2 - term3 + term4 - term5 - term6 - term7
            return ProfitTerm
        
        #Objective function
        obj_funct = (1-w)*Profit() 
        model.setObjective(obj_funct, GRB.MAXIMIZE)

        #%% Definition of contraints
       
        #Marketing Constraint 1
        
        for j in weeks:
            for k in bands:
                model.addConstr(
                    (gp.quicksum(dict_data['y_sijk'][scenario][i][j][k]*Hij[i,j] for i in crops) - Sjk[j,k] -Ojk[j,k]- gp.quicksum(Fjmk[j,m,k] for m in customers) )== 0,
                    f"Marketing Constraint 1 -  j: {j}, k: {k}"
                )



        #Marketing Constraint 2
        
        for j in weeks:
            for m in customers:
                for k in bands:
                    if(k not in dict_data["Km"][m]):
                        model.addConstr(
                        Fjmk[j,m,k] == 0,
                        f"Marketing Constraint 2 - j: {j}, m:{m}, k: {k}"
                        )


        #Demand Constraint 
        
        for j in weeks:
            for m in customers:
                model.addConstr(
                    (gp.quicksum(Fjmk[j,m,k] for k in dict_data["Km"][m])) == (dict_data['d_mj'][m][j] - Pmj[m,j]),
                    f"Demand Constraint - j: {j}, m: {m}"
                )



        #Sell on Open Market 

        for j in weeks:
                model.addConstr(
                    gp.quicksum(Sjk[j,k] for k in bands) <= 0.25*gp.quicksum(dict_data['d_mj'][m][j] for m in customers),
                    f" Sell on Open Market - j: {j}"
                )

  
 

        #Land Use Constraint - 2  
        for  i in crops:
            model.addConstr(
                ai[i] == gp.quicksum(Hij[i,j] for j in weeks)+OHi[i],
                f"Land Use Constraint - 2 -  i: {i}"
            )

        #Disease Constraint 
        for j in weeks:
            for q in diseases:
                model.addConstr(
                    gp.quicksum(dict_data["r_iq"][i][q]*gp.quicksum(dict_data['y_sijk'][scenario][i][j][k]*Hij[i,j] for k in bands) for i in crops) <= dict_data['u_q'][q]*gp.quicksum(dict_data['d_mj'][m][j] for m in customers),
                    f"Disease Constraint - j: {j}, q: {q}"
                )



     
        #%% Optimization of the model
        
        print("Gurobi model generation!")
        #Update model
        model.update()
        if verbose:
            model.setParam('OutputFlag', 1)
        else:
            model.setParam('OutputFlag', 0)
       

        
        print("Gurobi start!")
        start = time.time()
        model.optimize()
        model.write('C:\\Users\\Giulia\\Desktop\\PoliTO\\Operational research\\2021_ORTA_RobustPlantingPlans\\logs\\gurobi_secondStage.lp')
        end = time.time()
        comp_time = end - start

        print("Gurobi ended! = ", comp_time)
        
        #Preparation of results
        of = -1
        if model.status == GRB.Status.OPTIMAL:
            of = model.getObjective().getValue()
            
    
        return of, ai, comp_time


        

        
       
