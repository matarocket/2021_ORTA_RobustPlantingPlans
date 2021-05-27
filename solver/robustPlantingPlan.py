# -*- coding: utf-8 -*-
import time
import logging
import gurobipy as gp
from gurobipy import GRB


class SimpleKnapsack():
    def __init__(self):
        pass

    def solve(
        self, dict_data, reward, n_scenarios, time_limit=None,
        gap=None, verbose=False
    ):
        items = range(dict_data['scenarios'])
        scenarios = range(n_scenarios)

        problem_name = "RobustPlantingPlan"
        logging.info("{}".format(problem_name))
        # logging.info(f"{problem_name}")

        model = gp.Model(problem_name)
        Fsjmk = model.addVars(
            dict_data["scenarios"],dict_data["weeks"],dict_data["customers"],dict_data["bands"],
            lb=0,
            ub=1,
            vtype=GRB.INTEGER,
            name='Fsjmk'
        )
        Hsij = model.addVars(
            dict_data["scenarios"],dict_data['crops'],dict_data["weeks"],
            lb=0,
            ub=1,
            vtype=GRB.INTEGER,
            name='Hsij'
        )
        Ssjk = model.addVars(
                    dict_data["scenarios"],dict_data["weeks"],dict_data["bands"],
                    lb=0,
                    ub=1,
                    vtype=GRB.INTEGER,
                    name='Ssjk'
                )
        Lminus = model.addVars(
                    1,
                    lb=0,
                    ub=1,
                    vtype=GRB.CONTINUOUS,
                    name='Lminus'
                )

        Lplus = model.addVars(
                    1,
                    lb=0,
                    ub=1,
                    vtype=GRB.CONTINUOUS,
                    name='Lplus'
                )
        Psmj = model.addVars(
            dict_data["scenarios"],dict_data["customers"],dict_data["weeks"],
            lb=0,
            ub=1,
            vtype=GRB.INTEGER,
            name='Psmj'
        )
        Ai = model.addVars(
            dict_data['crops'],
            lb=0,
            ub=1,
            vtype=GRB.CONTINUOUS,
            name='Ai'
        )
   
        w=0.5
        
        #sum_s = gp.quicksum( gp.quicksum(dict_data["s_sj"][s][j] * Ssjk[s][j][k] for k in dict_data["bands"] )for j in dict_data["weeks"])
        #sum_f = gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fsjmk[s][j][m][k] for m in dict_data["customers"])for j in dict_data["weeks"] )for k in dict_data["bands"])
        #sum_cH = gp.quicksum( gp.quicksum(dict_data["c_sij"][s][i][j]*Hsij[s][i][j] for i in dict_data["crops"] )for j in dict_data["weeks"])
        #sum_cPrimeAi = gp.quicksum(dict_data["c_prime"]*Ai[i] for i in dict_data["crops"])
        #sum_p = gp.quicksum( gp.quicksum(dict_data["p_smj"][s][m][j]*Psmj[s][m][j] for m in dict_data["customers"] )for j in dict_data["weeks"])


        obj_funct = gp.quicksum((1-w)*dict_data["prob_s"][s] * (gp.quicksum( gp.quicksum(dict_data["s_sj"][s][j] * Ssjk[s][j][k] for k in dict_data["bands"] )for j in dict_data["weeks"])) for s in dict_data["scenarios"]+gp.quicksum(gp.quicksum( gp.quicksum(dict_data["f_mj"][m][j]*Fsjmk[s][j][m][k] for m in dict_data["customers"])for j in dict_data["weeks"] )for k in dict_data["bands"])-gp.quicksum( gp.quicksum(dict_data["c_sij"][s][i][j]*Hsij[s][i][j] for i in dict_data["crops"] )for j in dict_data["weeks"])+dict_data["c_plus"]*Lplus-dict_data["C_minus"]*Lminus-gp.quicksum(dict_data["c_prime"]*Ai[i] for i in dict_data["crops"])-gp.quicksum( gp.quicksum(dict_data["p_smj"][s][m][j]*Psmj[s][m][j] for m in dict_data["customers"] )for j in dict_data["weeks"]))
        # for s in scenarios:
        #     obj_funct += gp.quicksum(reward[i, s] * Y[i, s] for i in items)/(n_scenarios + 0.0)
        obj_funct += gp.quicksum(reward[i, s] * Y[i, s] for i in items for s in scenarios)/(n_scenarios + 0.0)
        model.setObjective(obj_funct, GRB.MAXIMIZE)

        model.addConstr(
            gp.quicksum(dict_data['sizes'][i] * X[i] for i in items) <= dict_data['max_size'],
            f"volume_limit_fs"
        )
        
        for s in scenarios:
            model.addConstr(
                gp.quicksum(dict_data['sizes_ss'][i] * Y[i, s] for i in items) <= dict_data['max_size_ss'],
                f"volume_limit_ss_{s}"
            )
        for i in items:
            model.addConstr(
                gp.quicksum(Y[i, s] for s in scenarios) <= n_scenarios * X[i],
                f"link_X_Y_for_item_{i}"
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

        start = time.time()
        model.optimize()
        end = time.time()
        comp_time = end - start
        
        sol = [0] * dict_data['n_items']
        of = -1
        if model.status == GRB.Status.OPTIMAL:
            for i in items:
                grb_var = model.getVarByName(
                    f"X[{i}]"
                )
                sol[i] = grb_var.X
            of = model.getObjective().getValue()
        return of, sol, comp_time
