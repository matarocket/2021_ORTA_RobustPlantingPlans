# -*- coding: utf-8 -*-
import numpy as np


# class Sampler:
#     def __init__(self):
#         pass

#     def sample_ev(self, instance, n_scenarios):
#         demand = self.sample_stoch(instance, n_scenarios)
#         return np.average(demand, axis=1)

#     def sample_stoch(self, instance, n_scenarios):
#         return np.around(np.absolute(np.random.normal(
#             10,
#             1,
#             size=(instance.n_items, n_scenarios))
#         ))

class Sampler:
    def __init__(self):
        pass

    def sample_stoch(self, instance):
        # return np.around(np.absolute(np.random.normal(
        #     10,
        #     1,
        #     size=(instance.n_items, n_scenarios))
        # ))
        prob_s = np.random.uniform(0, 1, instance.n_scenarios)
        prob_s = prob_s/np.sum(prob_s)
        
        return prob_s

    def sample_stoch_alternative(self, tot_scen, train_len):
        
        p=np.zeros(tot_scen-train_len)
        aux=[]
        for elem in p:
            aux.append(int(elem))
            
        prob_s = np.random.uniform(0, 1, train_len)
        prob_s = prob_s/np.sum(prob_s)
        prob=[j for i in [prob_s, aux] for j in i] 
        
        return prob
