# -*- coding: utf-8 -*-
import os
import time
import logging
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from simulator.instance import Instance
import heuristic.secondStageHeuristicGurobi as Heuristic


class Tester():
    def __init__(self):
        pass

    def compare_sols_lst(
        self, inst, sampler, sols, labels, n_scenarios
    ):
        ans_dict = {}
        reward = sampler.sample_stoch(
            inst,
            n_scenarios=n_scenarios
        )
        for j in range(len(sols)):
            profit_raw_data = self.solve_second_stages(
                inst, sols[j],
                n_scenarios, reward
            )
            ans_dict[labels[j]] = profit_raw_data

        return ans_dict

    def solve_second_stages(
        self, inst, sol, n_scenarios, reward
    ):
        ans = []
        obj_fs = 0
        for i in range(inst.n_items):
            obj_fs += inst.profits[i] * sol[i]
        items = range(inst.n_items)
        for s in range(n_scenarios):
            problem_name = "SecondStagePrb"
            model = gp.Model(problem_name)
            Y = model.addVars(
                inst.n_items,
                lb=0,
                ub=1,
                vtype=GRB.INTEGER,
                name='Y'
            )

            obj_funct = gp.quicksum(reward[i, s] * Y[i] for i in items)

            model.setObjective(obj_funct, GRB.MAXIMIZE)
            
            model.addConstr(
                gp.quicksum(inst.sizes_ss[i] * Y[i] for i in items) <= inst.max_size_ss,
                f"volume_limit_ss"
            )
            for i in items:
                model.addConstr(
                    Y[i] <= sol[i],
                    f"link_X_Y_for_item_{i}"
                )
            model.update()
            model.setParam('OutputFlag', 0)
            model.setParam('LogFile', './logs/gurobi.log')
            model.optimize()
            ans.append(obj_fs + model.getObjective().getValue())

        return ans

    # def in_sample_stability(self, problem, sampler, instance, n_repertions, n_scenarios_sol):
    #     ans = [0] * n_repertions
    #     for i in range(n_repertions):
    #         reward = sampler.sample_stoch(
    #             instance,
    #             n_scenarios=n_scenarios_sol
    #         )
    #         of, sol, comp_time = problem.solve(
    #             instance,
    #             reward,
    #             n_scenarios_sol
    #         )
    #         ans[i] = of
    #     return ans
    
    def in_sample_stability(self, sim_setting, problem, sampler, instance, dict_data, n_repertions, n_scenarios_sol):
        ans = [0] * n_repertions
        for i in range(n_repertions):
            
            sim_setting["n_scenarios"] = n_scenarios_sol
            inst = Instance(sim_setting)
            dict_data = inst.get_data()
            prob_s = sampler.sample_stoch(inst)
            
            of, _, _, _ = problem.solve(
                dict_data,
                prob_s
            )
            
            ans[i] = of
        return np.mean(ans), np.std(ans)
    
    def out_of_sample_stability(self, sim_setting, problem, sampler, instance, n_repertions, n_scenarios_sol, n_scenarios_out):
        ans = [0] * n_repertions
        heu1 = Heuristic.SecondStageSolver()
        
        for i in range(n_repertions):
            
            sim_setting["n_scenarios"] = n_scenarios_sol
            inst = Instance(sim_setting)
            dict_data = inst.get_data()
            prob_s = sampler.sample_stoch(inst)
            
            of, _, _, opt_model = problem.solve(
                dict_data,
                prob_s
            )
            
            crops = range(dict_data["crops"]) 
            L_minus = 0
            L_plus = 0
            A_i = np.zeros((dict_data['crops']))
            
            grb_var = opt_model.getVarByName("Lminus[0]") 
            L_minus = grb_var.X
            grb_var = opt_model.getVarByName("Lplus[0]") 
            L_plus = grb_var.X
            for i in crops:
                grb_var = opt_model.getVarByName(
                    f"Ai[{i}]"
                )
                A_i[i] = grb_var.X
            
            sim_setting["n_scenarios"] = n_scenarios_out
            inst = Instance(sim_setting)
            dict_data = inst.get_data()
            prob_s = sampler.sample_stoch(inst)
            
            profit = 0
            for s in n_scenarios_out:
                of_heu, sol_heu, comp_time = heu1.solve(dict_data, s, A_i, L_plus, L_minus)
                profit += of_heu*prob_s[s]
            
            ans[i] = profit
            
        return ans
