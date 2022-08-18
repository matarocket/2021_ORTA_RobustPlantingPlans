# -*- coding: utf-8 -*-
import logging
import numpy as np
import json
import matplotlib.pyplot as plt
import random

#%% INSTANCE GENERATION

#Instance class for a problem
class Instance():
     
    #Constructor
    def __init__(self, sim_setting):
        
        np.random.seed(0)
        
        #Log
        logging.info("starting simulation...")

        #Problem parameters
        self.n_diseases = sim_setting['n_diseases']
        self.n_varieties = sim_setting['n_varieties']
        self.n_spacings = sim_setting['n_spacings']
        self.n_size_bands = sim_setting['n_size_bands']
        self.n_customers = sim_setting['n_customers']
        self.n_scenarios = sim_setting['n_scenarios']
        self.n_sowing_dates = sim_setting['n_sowing_dates']
        self.n_harvesting_dates = sim_setting['n_harvesting_dates']
        self.w = sim_setting['w']

        #Number of crops
        self.n_crops = self.n_varieties*self.n_sowing_dates*self.n_spacings
        
        #Dictonary for identifying Crop characteristics
        self.Ai_dict = []
        for v in range(self.n_varieties):
            for w in range(self.n_sowing_dates):
                for s in range(self.n_spacings):
                    self.Ai_dict.append({"Variety" : v, "SowingDate": w, "Spacing": s})
                    
        #Variety growing rate
        self.variety_rate = np.random.normal(1, 0.1, self.n_varieties)
        
        #Vector of size bands preferences of customers
        self.Km_number = np.random.randint(1, self.n_size_bands, self.n_customers)
        self.Km = []
        for m in range(self.n_customers):
            self.Km.append(random.sample(list(range(self.n_size_bands)),self.Km_number[m]))
            
        
        #Probability of scenario s
        self.prob_s = np.random.uniform(0, 1, self.n_scenarios)
        self.prob_s = self.prob_s/np.sum(self.prob_s)
        
        #Impact of each scenario on yields
        self.scenario_impact = np.random.normal(1, 0.1, self.n_scenarios)
        
        #Cost of land - €/ha - checked for Italy
        #https://www.kpu.ca/sites/default/files/ISFS/Brussel%20Sprouts.pdf
        self.c_prime = np.random.normal(15000, 0.05*15000)
        
        #Cost of harvesting {scenario, crop, harvesting_date}
        #https://www.farmersjournal.ie/my-farming-christmas-anthony-weldon-swords-north-co-dublin-196397
        self.c_sij = np.random.normal(8700,0.01*8700,(self.n_scenarios, self.n_crops, self.n_harvesting_dates))
        
        #Demand {costumer, harvesting_date}
        #https://www.freshplaza.it/article/9397379/oltre-24mila-tonnellate-di-cavoletti-di-bruxelles-in-30-varieta-e-16-calibri-diversi/
        #demand proportioned to 10 customers 
        self.d_mj = np.random.normal(100, 0.35*100, (self.n_customers, self.n_harvesting_dates))
        
        #Profit {costumer, harvesting_date}
        #online supermarket 
        self.f_mj = np.random.normal(2000, 0.1*2000,(self.n_customers, self.n_harvesting_dates))
        
        #Surplus selling to the market {scenario, harvesting_date}
        #set to a half of the previous
        self.s_sj = np.random.normal(2000, 0.1*2000,(self.n_scenarios, self.n_harvesting_dates))
        
        #Yield crop {scenario, crop, harvesting_date, size_band}
        #https://www.agrifarming.in/brussels-sprout-cultivation-information
        self.y_sijk = np.random.rand(self.n_scenarios, self.n_crops, self.n_harvesting_dates, self.n_size_bands)
        self.yield_instance_gen()
        self.y_sijk = self.y_sijk*40
        
        #Area of grower's land
        self.a = 10000 
        
        #Cost of extra land required 
        #https://www.affittoterreno.com/prezzo-affitto-terreno-agricolo
        self.c_minus = np.random.randint(2000,3500) #+ self.c_prime
        
        #Cost of unused land
        #https://www.affittoterreno.com/prezzo-affitto-terreno-agricolo
        #since it is unused, you lost the price for the rental
        #the alternative could be to set 0 -> no use, nothing is lost 
        self.c_plus = np.random.randint(2000,3500)
        
        #Penalty of failiure in satisfying demand {scenario, costumer, harvesting_date}
        f_smj = np.zeros((self.n_scenarios, self.f_mj.shape[0], self.f_mj.shape[1]))
        for s in range(self.n_scenarios):
            f_smj[s,:,:] = np.copy(self.f_mj)
        self.f_smj = f_smj
            
        psmj = np.zeros((self.n_scenarios, self.n_customers, self.n_harvesting_dates))
        for s in range(self.n_scenarios):
            psmj[s] = self.f_mj
        self.p_smj = psmj
        
        #Susceptibility to diseases {crop, disease}
        self.r_iq = np.around(np.random.rand(self.n_crops,self.n_diseases), 0)
        
        #Upper limit to harvested proportion of a field for a given disease
        self.u_q = np.random.uniform(0.7,1,self.n_diseases)

        #Log variables
        logging.info(f"crops: {self.n_crops}")
        logging.info(f"varieties: {self.n_varieties}")
        logging.info(f"varieties_rates: {self.variety_rate}")
        logging.info(f"weeks: {self.n_harvesting_dates}")
        logging.info(f"spacings: {self.n_spacings}")
        logging.info(f"bands: {self.n_size_bands}")
        logging.info(f"customers: {self.n_customers}")
        logging.info(f"diseases: {self.n_diseases}")
        logging.info(f"scenarios: {self.n_scenarios}")
        logging.info(f"scenarios_impact: {self.scenario_impact}")
        logging.info(f"prob s: {self.prob_s}")
        logging.info(f"w: {self.w}")
        logging.info(f"c prime: {self.c_prime}")
        logging.info(f"c_sij: {self.c_sij}")
        logging.info(f"d_mj: {self.d_mj}")
        logging.info(f"f_mj: {self.f_mj}")
        logging.info(f"s_sj: {self.s_sj}")
        logging.info(f"y_sijk: {self.y_sijk}")
        logging.info(f"a: {self.a}")
        logging.info(f"c_minus: {self.c_minus}")
        logging.info(f"c_plus: {self.c_plus}")
        logging.info(f"p_smj: {self.p_smj}")
        logging.info(f"r_iq: {self.r_iq}")
        logging.info(f"u_q: {self.u_q}")
        logging.info(f"Km: {self.Km}")
        logging.info(f"Ai_dict: {self.Ai_dict}")
        logging.info("simulation end")
        
        return
        
    #Yield instance generation  
    def yield_instance_gen(self):
        for s in range(self.n_scenarios):
            for i in range(self.n_crops):
                for j in range(self.n_harvesting_dates):
                    for k in range(self.n_size_bands):
                        #print(s, " , ",i, " , ",j, " , ",k)
                        #self.y_sijk[s][i][j][k] = self.scenario_impact[s]*grow_week_curve(self.Ai_dict[i]["Variety"], j + (self.Ai_dict[i]["SowingDate"] - (self.n_sowing_dates-1)))
                        aux_rate = self.variety_rate[self.Ai_dict[i]["Variety"]]
                        #TODO fix grow_week_curve -> it should became Gaussian
                        self.y_sijk[s][i][j][k] = (k+1)*self.scenario_impact[s]*grow_week_curve(aux_rate*(k+1), j + ((self.n_sowing_dates-1) - self.Ai_dict[i]["SowingDate"])) 
                        #print(self.y_sijk[s][i][j][k])
                        pass
        
    #Get data (instance as dictionary)
    def get_data(self):
        logging.info("getting data from instance...")
        return {
            "crops": self.n_crops,
            "varieties": self.n_varieties,
            "weeks": self.n_harvesting_dates,
            "bands": self.n_size_bands,
            "customers": self.n_customers,
            "diseases": self.n_diseases,
            "scenarios": self.n_scenarios,
            "w" : self.w,
            "prob_s": self.prob_s,
            "c_prime": self.c_prime,
            "c_sij": self.c_sij,
            "d_mj": self.d_mj,
            "f_mj": self.f_mj,
            "s_sj": self.s_sj,
            "y_sijk": self.y_sijk,
            "a": self.a,
            "c_minus": self.c_minus,
            "c_plus": self.c_plus,
            "p_smj": self.p_smj,
            "r_iq": self.r_iq,
            "u_q": self.u_q,
            "Km": self.Km,
            "Ai_dict" : self.Ai_dict,
            "sowingWeeks": self.n_sowing_dates,
            "spacings": self.n_spacings,
        }
    
    #Plot crop evolution
    def plot_crop_evolution(self, crop_index=0, scene_idx=0):
        crop=crop_index
        print(self.Ai_dict[crop])
        for k in range(self.n_size_bands):
            plt.plot(self.y_sijk[scene_idx,crop,:,k])
        plt.show()

#Poisson distribution
def poisson_dist(k, mu):
    if(k >= 0):
        y = np.exp(-mu)*(mu**k)/np.math.factorial(k) 
    else:
        y = 0
    return y

#Growing week curve
def grow_week_curve(grow_factor, week):
    x = week
    #y = grow_factor*x*np.exp(-x)
    y = poisson_dist(x, grow_factor)
    if(y<=0):
        y=0
    return y



#Testing main
if __name__ == "__main__":
    
    #Load conf file
    fp = open("../etc/sim_setting.json", 'r')
    sim_setting = json.load(fp)
    fp.close()
    
    #Generate an instance
    ins = Instance(sim_setting)
    #ins.yield_instance_gen()
    
    #Plot the evolution of a certain crop
    ins.plot_crop_evolution(0)
