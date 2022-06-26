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
            self, dict_data, reward=None,time_limit=None, 
            gap=None, verbose=False
            ):
        
        scenarios = range(dict_data['scenarios'])
        weeks = range(dict_data['weeks'])
        bands = range(dict_data['bands'])
        customers = range(dict_data["customers"]) 
        crops = range(dict_data["crops"]) 
        diseases = range(dict_data["diseases"]) 
        varieties = range(dict_data["varieties"]) 

        problem_name = "RobustPlantingPlan"
        logging.info("{}".format(problem_name))
        # logging.info(f"{problem_name}")

        model = gp.Model(problem_name)

        #model variable definition 
        Fsjmk = model.addVars(
            dict_data["scenarios"],dict_data["weeks"],dict_data["customers"],dict_data["bands"],
            lb=0,
            ub=10000,
            vtype=GRB.CONTINUOUS,
            name='Fsjmk'
        )
        Hsij = model.addVars(
            dict_data["scenarios"],dict_data['crops'],dict_data["weeks"],
            lb=0,
            ub=10000,
            vtype=GRB.CONTINUOUS,
            name='Hsij'
        )
        Ssjk = model.addVars(
                    dict_data["scenarios"],dict_data["weeks"],dict_data["bands"],
                    lb=0,
                    ub=10000,
                    vtype=GRB.CONTINUOUS,
                    name='Ssjk'
                )
        Lminus = model.addVars(
                    1,
                    lb=0,
                    ub=10000,
                    vtype=GRB.CONTINUOUS,
                    name='Lminus'
                )

        Lplus = model.addVars(
                    1,
                    lb=0,
                    ub=10000,
                    vtype=GRB.CONTINUOUS,
                    name='Lplus'
                )
        Psmj = model.addVars(
            dict_data["scenarios"],dict_data["customers"],dict_data["weeks"],
            lb=0,
            ub=10000,
            vtype=GRB.CONTINUOUS,
            name='Psmj'
        )
        Ai = model.addVars(
            dict_data['crops'],
            lb=0,
            ub=10000,
            vtype=GRB.CONTINUOUS,
            name='Ai'
        )
   
        w=0.5

        z = model.addVars(
            dict_data["scenarios"],
            lb=0,
            ub=100,
            vtype=GRB.CONTINUOUS,
            name='z'
        )
        
        #sum_s = gp.quicksum( gp.quicksum(dict_data["s_sj"][s][j] * Ssjk[s][j][k] for k in dict_data["bands"] )for j in dict_data["weeks"])
        #sum_f = gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fsjmk[s][j][m][k] for m in dict_data["customers"])for j in dict_data["weeks"] )for k in dict_data["bands"])
        #sum_cH = gp.quicksum( gp.quicksum(dict_data["c_sij"][s][i][j]*Hsij[s][i][j] for i in dict_data["crops"] )for j in dict_data["weeks"])
        #sum_cPrimeAi = gp.quicksum(dict_data["c_prime"]*Ai[i] for i in dict_data["crops"])
        #sum_p = gp.quicksum( gp.quicksum(dict_data["p_smj"][s][m][j]*Psmj[s][m][j] for m in dict_data["customers"] )for j in dict_data["weeks"])

        #objective function definition
        #profit term

        #E_ProfitTerm = gp.quicksum((1-w)*dict_data["prob_s"][s] * (gp.quicksum( gp.quicksum(dict_data["s_sj"][s][j] * Ssjk[s][j][k] for k in dict_data["bands"] )for j in dict_data["weeks"])) for s in dict_data["scenarios"]+gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fsjmk[s][j][m][k] for m in dict_data["customers"])for j in dict_data["weeks"] )for k in dict_data["bands"])-gp.quicksum( gp.quicksum(dict_data["c_sij"][s][i][j]*Hsij[s][i][j] for i in dict_data["crops"] )for j in dict_data["weeks"])+dict_data["c_plus"]*Lplus-dict_data["C_minus"]*Lminus-gp.quicksum(dict_data["c_prime"]*Ai[i] for i in dict_data["crops"])-gp.quicksum( gp.quicksum(dict_data["p_smj"][s][m][j]*Psmj[s][m][j] for m in dict_data["customers"] )for j in dict_data["weeks"]))
        #obj_funct = gp.quicksum((1-w)*dict_data["prob_s"][s] * (gp.quicksum( gp.quicksum(dict_data["s_sj"][s][j] * Ssjk[s][j][k] for k in bands )for j in weeks)) for s in scenarios+gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fsjmk[s][j][m][k] for m in customers)for j in weeks )for k in bands)-gp.quicksum( gp.quicksum(dict_data["c_sij"][s][i][j]*Hsij[s][i][j] for i in crops )for j in weeks)+dict_data["c_plus"]*Lplus-dict_data["C_minus"]*Lminus-gp.quicksum(dict_data["c_prime"]*Ai[i] for i in crops)-gp.quicksum( gp.quicksum(dict_data["p_smj"][s][m][j]*Psmj[s][m][j] for m in customers )for j in weeks))
        
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
        
        
        
        
            
        
        #E_ProfitTerm = gp.quicksum(dict_data["prob_s"][s] * (gp.quicksum( gp.quicksum(dict_data["s_sj"][s][j] * Ssjk[s][j][k] for k in bands )for j in weeks)) +gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fsjmk[s][j][m][k] for m in customers)for j in weeks )for k in bands)-gp.quicksum( gp.quicksum(dict_data["c_sij"][s][i][j]*Hsij[s][i][j] for i in crops )for j in weeks)+dict_data["c_plus"]*Lplus-dict_data["C_minus"]*Lminus-gp.quicksum(dict_data["c_prime"]*Ai[i] for i in crops)-gp.quicksum( gp.quicksum(dict_data["p_smj"][s][m][j]*Psmj[s][m][j] for m in customers )for j in weeks)for s in scenarios)
        
        
       #min z to max the obj funct
        #aux = gp.quicksum(dict_data["prob_s"][s] * (ProfitTerm - E_ProfitTerm) for s in scenarios)
        
        #z>=aux
        #z>=-aux


        obj_funct=((1-w)*E_s(Profit) - w*gp.quicksum(dict_data["prob_s"][s]*z[s] for s in scenarios))
        #obj_funct=E_s(Profit)
        #missing risk term - to be added 
        #riskTerm =
        #obj_funct = gp.quicksum((1-w)*dict_data["prob_s"][s] * (gp.quicksum( gp.quicksum(dict_data["s_sj"][s][j] * Ssjk[s][j][k] for k in dict_data["bands"] )for j in dict_data["weeks"])) for s in dict_data["scenarios"]+gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fsjmk[s][j][m][k] for m in dict_data["customers"])for j in dict_data["weeks"] )for k in dict_data["bands"])-gp.quicksum( gp.quicksum(dict_data["c_sij"][s][i][j]*Hsij[s][i][j] for i in dict_data["crops"] )for j in dict_data["weeks"])+dict_data["c_plus"]*Lplus-dict_data["C_minus"]*Lminus-gp.quicksum(dict_data["c_prime"]*Ai[i] for i in dict_data["crops"])-gp.quicksum( gp.quicksum(dict_data["p_smj"][s][m][j]*Psmj[s][m][j] for m in dict_data["customers"] )for j in dict_data["weeks"]))
        
        # for s in scenarios:
        #     obj_funct += gp.quicksum(reward[i, s] * Y[i, s] for i in items)/(n_scenarios + 0.0)

        #putting together risk and profit terms 
        #obj_funct += gp.quicksum(reward[i, s] * Y[i, s] for i in items for s in scenarios)/(n_scenarios + 0.0)
        model.setObjective(obj_funct, GRB.MAXIMIZE)


        #------------------- Constraints definition -------------------
       
        #>>> Absolute value Constraint
        for s in scenarios:
            model.addConstr(
                        (z[s] - (Profit(s) - E_s(Profit))) >= 0,
                        f"Absolute value of z 1 - s: {s}"
                    )
            model.addConstr(
                        (z[s] + (Profit(s) - E_s(Profit))) >= 0,
                        f"Absolute value of z 2 - s: {s}"
                    )
        
       
        #>>> Marketing Constraint 
        for s in scenarios:
            for j in weeks:
                for k in bands:
                    model.addConstr(
                        gp.quicksum(dict_data['y_sijk'][s][i][j][k]*Hsij[s,i,j] for i in crops)-Ssjk[s,j,k]-gp.quicksum(Fsjmk[s,j,m,k] for m in customers)==0,
                        f"Marketing Constraint - s: {s}, j: {j}, k: {k}"
                    )


        #>>> Demand Constraint 
        for s in scenarios:
            for j in weeks:
                for m in customers:
                    model.addConstr(
                        gp.quicksum(Fsjmk[s,j,m,k] for k in dict_data["Km"]) == Psmj[s,m,j] + dict_data['d_mj'][m][j],
                        f"Demand Constraint - s: {s}, j: {j}, m: {m}"
                    )


        #>>> Sell on Open Market 
        for s in scenarios:
            for j in weeks:
                    model.addConstr(
                        gp.quicksum(Ssjk[s,j,k] for k in bands) <= 0.25*gp.quicksum(dict_data['d_mj'][m][j] for m in customers),
                        f" Sell on Open Market - s: {s}, j: {j}"
                    )
                

        #>>> Land Use Constraint - 1  
        model.addConstr(
                    gp.quicksum(Ai[i] for i in crops) == dict_data["a"]+ Lminus[0]-Lplus[0] ,
                    f"Land Use Constraint - 1"
                )


        #>>> Land Use Constraint - 2  
        for s in scenarios:
            for  i in crops:
                model.addConstr(
                    Ai[i]== gp.quicksum(Hsij[s,i,j] for j in weeks),
                    f"Land Use Constraint - 2 - s: {s}, i: {i}"
                )


        #>>> Disease Constraint 
        for j in weeks:
            for q in diseases:
                for s in scenarios:
                    model.addConstr(
                        gp.quicksum(dict_data["r_iq"][i][q]*gp.quicksum(dict_data['y_sijk'][s][i][j][k]*Hsij[s,i,j] for k in bands) for i in crops) <= dict_data['u_q'][q]*gp.quicksum(dict_data['d_mj'][m][j] for m in customers),
                        f"Disease Constraint - s: {s}, j: {j}, q: {q}"
                    )


        #>>> Individual Variety Limit 
        
        for v in varieties:
            aux=[]
            for ind in range(len(crops)):
                if (int(ind / (dict_data["sowingWeeks"]*dict_data["spacings"])) == v):
                    aux.append(crops[ind])
                # if (dict_data["Ai_dict"][ind]["Variety"] == v):
                #     aux.append(crops[ind])
            model.addConstr(
                gp.quicksum(Ai[index] for index in aux) <= 0.4*gp.quicksum(Ai[i] for i in crops),
                f"Individual Variety Limit - v: {v}"
            )

        


        # #>>> Individual Crop Limit 
        for i in crops:
                    model.addConstr(
                        Ai[i]<= 0.2*gp.quicksum(Ai[i] for i in crops),
                        f"Individual Crop Limit - i: {i}"
                    )
        

        

        model.update()
        if gap:
            model.setParam('MIPgap', gap)
        if time_limit:
            model.setParam(GRB.Param.TimeLimit, time_limit)
        if verbose:
            model.setParam('OutputFlag', 1)
        else:
            model.setParam('OutputFlag', 0)
        model.setParam('LogFile', './logs/gurobi.log')
        # model.write("./logs/model.lp")

        #Measure of execution time
        start = time.time()
        model.optimize()
        end = time.time()
        comp_time = end - start
        
        #Preparation of results
        sol = [0] * dict_data['crops']
        of = -1
        if model.status == GRB.Status.OPTIMAL:
            for i in crops:
                grb_var = model.getVarByName(
                    f"Ai[{i}]"
                )
                sol[i] = grb_var.X
            of = model.getObjective().getValue()
        return of, sol, comp_time, model
