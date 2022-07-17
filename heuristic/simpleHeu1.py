# -*- coding: utf-8 -*-
import time
import numpy as np
import gurobipy as gp
from gurobipy import GRB

class ProgressiveHedging():
    """Class representing the PH heuristics method.
    It has two methods:
        1. DEP_solver() to solve the deterministic problem with Augmented Relaxation
        2. solve() is the actual Progressive Hedging algorithm
    """
    def __init__(self):
        pass

    def DEP_solver(self, dict_data, scenario_input, TGS, lambd=0, pen_rho=0, iteration=0):
        problem_name = "DEP"
        model = gp.Model(problem_name)
        n_stations = dict_data['crops']
        stations = range(n_stations)
                
        ### Variables
        #Weight of sprouts sold by {scenario, week, customer, band}
        Fsjmk = model.addVars(
                dict_data["scenarios"],dict_data["weeks"],dict_data["customers"],dict_data["bands"],
                lb=0,
                #ub=10000,
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
                ub=dict_data["a"]/10,
                vtype=GRB.CONTINUOUS,
                name='Lminus'
            )

        #Area of unused land
        Lplus = model.addVars(
                1,
                lb=0,
                ub=dict_data["a"]/10,
                vtype=GRB.CONTINUOUS,
                name='Lplus'
            )
        
        #Shortage in demand by {scenario, customer, week}
        Psmj = model.addVars(
                dict_data["scenarios"],dict_data["customers"],dict_data["weeks"],
                lb=0,
                #ub=10000,
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

        #Auxiliar variable by {scenario}
        z = model.addVars(
                dict_data["scenarios"],
                #lb=0,
                vtype=GRB.CONTINUOUS,
                name='z'
            )
        
        model.update()
        scenarios = range(dict_data['scenarios'])
        weeks = range(dict_data['weeks'])
        bands = range(dict_data['bands'])
        customers = range(dict_data["customers"]) 
        crops = range(dict_data["crops"]) 
        diseases = range(dict_data["diseases"]) 
        varieties = range(dict_data["varieties"]) 
        w = dict_data["w"]
         #Preparation of results
        sol = [0] * dict_data['crops']
        if model.status == GRB.Status.OPTIMAL:
            for i in crops:
                grb_var = model.getVarByName(
                    f"Ai[{i}]"
                )
                sol[i] = grb_var.X


        ## Objective Function
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
            Expected = gp.quicksum(dict_data["prob_s"][s]*function_of_s(s) for s in scenarios)
            return Expected
        
        #Objective function
        obj_funct = (1-w)*E_s(Profit) - w*gp.quicksum(dict_data["prob_s"][s]*z[s] for s in scenarios)
        model.setObjective(obj_funct, GRB.MAXIMIZE)
        
        if iteration!= 0:
            relax = np.dot(lambd.T,(np.array(sol)-TGS))
            penalty = (pen_rho/2)*(np.dot((np.array(sol)-TGS),(np.array(sol)-TGS).T))
            
            obj_funct += relax
        
            obj_funct += penalty


        model.setObjective(obj_funct, GRB.MINIMIZE)


        ### Costraints

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
                        gp.quicksum(dict_data['y_sijk'][s][i][j][k]*Hsij[s,i,j] for i in crops) - Ssjk[s,j,k] - gp.quicksum(Fsjmk[s,j,m,k] for m in customers) == 0,
                        f"Marketing Constraint - s: {s}, j: {j}, k: {k}"
                    )

        #Demand Constraint 
        for s in scenarios:
            for j in weeks:
                for m in customers:
                    model.addConstr(
                        gp.quicksum(Fsjmk[s,j,m,k] for k in dict_data["Km"][m]) == Psmj[s,m,j] + dict_data['d_mj'][m][j],
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

        #Land Use Constraint - 2  
        for s in scenarios:
            for  i in crops:
                model.addConstr(
                    Ai[i] == gp.quicksum(Hsij[s,i,j] for j in weeks),
                    f"Land Use Constraint - 2 - s: {s}, i: {i}"
                )

        #Disease Constraint 
        for j in weeks:
            for q in diseases:
                for s in scenarios:
                    model.addConstr(
                        gp.quicksum(dict_data["r_iq"][i][q]*gp.quicksum(dict_data['y_sijk'][s][i][j][k]*Hsij[s,i,j] for k in bands) for i in crops) <= dict_data['u_q'][q]*gp.quicksum(dict_data['d_mj'][m][j] for m in customers),
                        f"Disease Constraint - s: {s}, j: {j}, q: {q}"
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

        model.update()
        model.setParam('OutputFlag', 0)
        model.optimize()
        sol = [0] * dict_data['crops']
        of = -1
        if model.status == GRB.Status.OPTIMAL:
            for i in stations:
                grb_var = model.getVarByName(
                    f"X[{i}]"
                )
                sol[i] = grb_var.X
            of = model.getObjective().getValue()
        return of, np.array(sol)
    

    def solve(
        self, instance, scenarios, n_scenarios, rho = 70, alpha=100
    ):
        ans = []
        of_array = []
        dict_data = instance.get_data()

        # temporary global solution (initialize the TGS for the first iteration)
        TGS = 0

        # max iterations
        maxiter = 100

        #iteration
        k=0

        #lagrangian multiplier
        lam = np.zeros((n_scenarios, dict_data['crops']))
        start = time.time()
        # solve the base problem for each of the solutions 
        # (solve the problem for the first time in order to initialize the first stage solution at the zero-th iteration)
        # For each scenario, solve the mono-scenario problem
        for i, s in enumerate(np.rollaxis(n_scenarios, 2)):
            of, sol = self.DEP_solver(dict_data, s, TGS, lam[i], rho)
            of_array.append(of)
            ans.append(np.array(sol))
        x_s_arrays = ans
        
        # compute temporary global solution for first iteration
        TGS = np.average(x_s_arrays, axis=0).astype(int)
        lam = rho*(x_s_arrays - TGS)

        for k in range(1, maxiter+1):

            if ( np.all(abs(x_s_arrays-TGS) == 0) ):
                break

   
            x_s_arrays = []
            of_array = []
            ans = []

            # solve monoscenario problems
            for i, s in enumerate(np.rollaxis(scenarios, 2)):
                of, sol = self.DEP_solver(dict_data, s, TGS, lam[i], rho, k)
                of_array.append(of)
                ans.append(np.array(sol))
            x_s_arrays = ans

            # compute temporary global solution for first iteration
            TGS = np.average(x_s_arrays, axis=0).astype(int)
            

            # update the multipliers
            lam = lam + rho*(x_s_arrays - TGS)
            rho = alpha*rho

        end = time.time()
        comp_time = end - start

        sol_x = TGS
        of = np.average(of_array, axis=0)
        return of, sol_x, comp_time, k