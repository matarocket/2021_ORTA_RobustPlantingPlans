# -*- coding: utf-8 -*-
import numpy as np

class Sampler:
    def __init__(self):
        pass

    #Sample probabilities for each scenario 
    def sample_stoch(self, instance):
        
        prob_s = np.random.uniform(0, 1, instance.n_scenarios)
        prob_s = prob_s/np.sum(prob_s)
        
        return prob_s

