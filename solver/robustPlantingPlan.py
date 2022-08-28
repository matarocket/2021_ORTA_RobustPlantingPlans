# -*- coding: utf-8 -*-
from calendar import week
import time
import logging
import gurobipy as gp
from gurobipy import GRB

class RobustPlantingPlanSolver():
    def __init__(self):
        pass

    def solve(
            self, dict_data, prob_s=None,time_limit=None, 
            gap=None, verbose=False
            ):
        
        
        #Load data from configuration file
        scenarios = range(dict_data['scenarios'])
        weeks = range(dict_data['weeks'])
        bands = range(dict_data['bands'])
        customers = range(dict_data["customers"]) 
        crops = range(dict_data["crops"]) 
        diseases = range(dict_data["diseases"]) 
        varieties = range(dict_data["varieties"]) 
        w = dict_data["w"]

        #Problem name logging
        problem_name = "RobustPlantingPlan"
        logging.info("{}".format(problem_name))
        # logging.info(f"{problem_name}")

        #Creation of model
        model = gp.Model(problem_name)

        #%% Control variables definition
        
        #Weight of sprouts sold by {scenario, week, customer, band}
        Fsjmk = model.addVars(
                dict_data["scenarios"],dict_data["weeks"],dict_data["customers"],dict_data["bands"],
                vtype=GRB.CONTINUOUS,
                name='Fsjmk'
            )
        
        #Area of harvested crop by {scenario, crop, week}
        Hsij = model.addVars(
                dict_data["scenarios"],dict_data['crops'],dict_data["weeks"],
                lb=0,
                #ub=10000,
                vtype=GRB.CONTINUOUS,
                name='Hsij'
            )
        
        #Weight of surplus-to-demand sprouts by {scemario, week, band}
        Ssjk = model.addVars(
                dict_data["scenarios"],dict_data["weeks"],dict_data["bands"],
                lb=0,
                #ub=10000,
                vtype=GRB.CONTINUOUS,
                name='Ssjk'
            )
        
        #Area of extra land 
        Lminus = model.addVars(
                1,
                lb=0,
                ub=dict_data["a"],
                vtype=GRB.CONTINUOUS,
                name='Lminus'
            )

        #Area of unused land
        Lplus = model.addVars(
                1,
                lb=0,
                ub=dict_data["a"],
                vtype=GRB.CONTINUOUS,
                name='Lplus'
            )
        
        #Shortage in demand by {scenario, customer, week}
        Psmj = model.addVars(
                dict_data["scenarios"],dict_data["customers"],dict_data["weeks"],
                #lb=0,
                #ub=300,
                vtype=GRB.CONTINUOUS,
                name='Psmj'
            )
        
        #Area of sowed land by {crop}
        Ai = model.addVars(
                dict_data['crops'],
                lb=0,
                ub=dict_data["a"],
                vtype=GRB.CONTINUOUS,
                name='Ai'
            )

        #over harvesting
        Trash= model.addVars(
            dict_data["crops"], 
            #lb=0,
            #ub=10000,
            vtype=GRB.CONTINUOUS,
            name='Trash'
        )

        #Auxiliar variable by {scenario}
        z = model.addVars(
                dict_data["scenarios"],
                #lb=0,
                vtype=GRB.CONTINUOUS,
                name='z'
            )
        
        #%% Definition of the cost function
        
        #Profit for a given scenario
        def Profit(s):
            term1 = (gp.quicksum( gp.quicksum(dict_data["s_sj"][s][j] * Ssjk[s,j,k] for j in weeks )for k in bands))
            term2 = gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fsjmk[s,j,m,k] for m in customers)for j in weeks )for k in bands)
            term3 = gp.quicksum( gp.quicksum(dict_data["c_sij"][s][i][j]*Hsij[s,i,j] for i in crops )for j in weeks)
            term4 = dict_data["c_plus"]*Lplus[0]
            term5 = dict_data["c_minus"]*Lminus[0]
            term6 = gp.quicksum(dict_data["c_prime"]*Ai[i] for i in crops)
            term7 = gp.quicksum( gp.quicksum(dict_data["p_smj"][s][m][j]*Psmj[s,m,j] for m in customers )for j in weeks)
            ProfitTerm =  term1 + term2 - term3 + term4 - term5 - term6 - term7
            return ProfitTerm
        
        #Expected value for a function in s
        def E_s(function_of_s):
            Expected = gp.quicksum(prob_s[s]*function_of_s(s) for s in scenarios)
            return Expected
        
        #Objective function
        obj_funct = (1-w)*E_s(Profit) - w*gp.quicksum(prob_s[s]*z[s] for s in scenarios)
        model.setObjective(obj_funct, GRB.MAXIMIZE)
        

        #%% Definition of contraints
       
        #Absolute value Constraint
        for s in scenarios:
            model.addConstr(
                        (z[s] - (Profit(s) - E_s(Profit))) >= 0,
                        f"Absolute value of z 1 - s: {s}"
                    )
            model.addConstr(
                        (z[s] + (Profit(s) - E_s(Profit))) >= 0,
                        f"Absolute value of z 2 - s: {s}"
                    )
        
        #Marketing Constraint 
        for s in scenarios:
            for j in weeks:
                for k in bands:
                    model.addConstr(
                        (gp.quicksum(dict_data['y_sijk'][s][i][j][k]*Hsij[s,i,j] for i in crops) - Ssjk[s,j,k] - gp.quicksum(Fsjmk[s,j,m,k] for m in customers) )== 0,
                        f"Marketing Constraint - s: {s}, j: {j}, k: {k}"
                    )



        #Vu cumprà Constraint
        for s in scenarios:
            for j in weeks:
                for m in customers:
                    for k in bands:
                        if(k not in dict_data["Km"][m]):
                            model.addConstr(
                            Fsjmk[s,j,m,k] == 0,
                            f"Vu cumprà Constraint - s: {s}, j: {j}, m:{m}, k: {k}"
                            )


        #Demand Constraint 
        for s in scenarios:
            for j in weeks:
                for m in customers:
                    model.addConstr(
                        #gp.quicksum(Fsjmk[s,j,m,k] for k in bands) == Psmj[s,m,j] + dict_data['d_mj'][m][j],
                        (gp.quicksum(Fsjmk[s,j,m,k] for k in dict_data["Km"][m])) == (dict_data['d_mj'][m][j] - Psmj[s,m,j]),
                        #Fsjmk[s,j,m,dict_data["Km"][m]] == Psmj[s,m,j] + dict_data['d_mj'][m][j],
                        f"Demand Constraint - s: {s}, j: {j}, m: {m}"
                    )



        #Sell on Open Market 
        for s in scenarios:
            for j in weeks:
                    model.addConstr(
                        gp.quicksum(Ssjk[s,j,k] for k in bands) <= 0.25*gp.quicksum(dict_data['d_mj'][m][j] for m in customers),
                        f" Sell on Open Market - s: {s}, j: {j}"
                    )
                
        #Land Use Constraint - 1  
        model.addConstr(
                    gp.quicksum(Ai[i] for i in crops) == dict_data["a"] + Lminus[0] - Lplus[0] ,
                    "Land Use Constraint - 1"
                )


        #Disease Constraint 
        for s in scenarios:
            for i in crops:
                model.addConstr(
                    Trash[i] == gp.quicksum(dict_data["r_iq"][i][q]*dict_data["u_q"][q]*Ai[i] for q in diseases),
                    f"Disease Constraint - i: {i} "
                )


        #Land Use Constraint - 2  
        for s in scenarios:
            for  i in crops:
                model.addConstr(
                    Ai[i] == gp.quicksum(Hsij[s,i,j] for j in weeks)+Trash[i],
                    f"Land Use Constraint - 2 - s: {s}, i: {i}"
                )
        
        
                

        #Individual Variety Limit 
        for v in varieties:
            aux=[]
            for i in crops:
                 if (dict_data["Ai_dict"][i]["Variety"] == v):
                     aux.append(i)
            model.addConstr(
                gp.quicksum(Ai[index] for index in aux) <= 0.4*gp.quicksum(Ai[i] for i in crops),
                f"Individual Variety Limit - v: {v}"
            )

        #Individual Crop Limit 
        for i in crops:
                    model.addConstr(
                        Ai[i] <= 0.2*gp.quicksum(Ai[i_2] for i_2 in crops),
                        f"Individual Crop Limit - i: {i}"
                    )
        
        #%% Optimization of the model
        
        print("Gurobi model generation!")

        #Update model
        model.update()
        if gap:
            model.setParam('MIPgap', gap)
        if time_limit:
            model.setParam(GRB.Param.TimeLimit, time_limit)
        if verbose:
            model.setParam('OutputFlag', 1)
        else:
            model.setParam('OutputFlag', 0)
        model.setParam('LogFile', 'C:\\Users\\Giulia\\Desktop\\PoliTO\\Operational research\\2021_ORTA_RobustPlantingPlans\\logs\\gurobi.log')
        # model.write("./logs/model.lp")

        
        print("Gurobi start!")

        start = time.time()
        model.optimize()
        model.write('C:\\Users\\Giulia\\Desktop\\PoliTO\\Operational research\\2021_ORTA_RobustPlantingPlans\\logs\\gurobi_optimal.lp')
        end = time.time()
        comp_time = end - start
        
       
        print("Gurobi ended! = ", comp_time)
        
        #Preparation of results
        sol = [0] * dict_data['crops']
        surplus = [0] * dict_data["bands"]
        of = -1
        if model.status == GRB.Status.OPTIMAL:
            for i in crops:
                grb_var = model.getVarByName(f"Ai[{i}]")
                sol[i] = grb_var.X
            of = model.getObjective().getValue()
        
        return of, sol, comp_time, model
