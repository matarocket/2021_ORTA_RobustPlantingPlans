# -*- coding: utf-8 -*-
import logging
import numpy as np

def randGeneratorCostumer(K, M):
        np.random.seed(1)
        vet=np.random.randint(1, K, M)
        return vet

class Instance():
     
        
    def __init__(self, sim_setting):
        logging.info("starting simulation...")

        self.sowingWeeks=sim_setting['sowingWeeks_max_index']
        self.spacings= sim_setting['spacings_max_index']
        self.crops = sim_setting['variety_max_index']*sim_setting['sowingWeeks_max_index']*sim_setting['spacings_max_index']
        self.varieties = sim_setting['variety_max_index']
        self.weeks = sim_setting['week_max_index']
        self.bands = sim_setting['size_band_max_index']
        self.customers = sim_setting['customer_max_index']
        self.diseases = sim_setting['disease_max_index']
        self.scenarios = sim_setting['scenarios_max_index']

        self.Km = randGeneratorCostumer(self.bands, self.customers)

        self.prob_s = np.around(np.random.uniform(0,1,sim_setting['scenarios_max_index']), 2)
        self.prob_s = self.prob_s/np.sum(self.prob_s)
        self.c_prime=np.random.randint(100,100000)
        self.c_sij=np.random.rand(sim_setting['scenarios_max_index'],sim_setting["crop_max_index"],sim_setting['week_max_index'])*100
        self.d_mj=np.random.rand(sim_setting['customer_max_index'],sim_setting["week_max_index"])*10
        self.f_mj=np.random.rand(sim_setting['customer_max_index'],sim_setting["week_max_index"])*50
        self.s_sj=np.random.rand(sim_setting['scenarios_max_index'],sim_setting["week_max_index"])*25
        self.y_sijk=np.random.rand(sim_setting['scenarios_max_index'],sim_setting["crop_max_index"],sim_setting["week_max_index"],sim_setting['size_band_max_index'],)*5
        self.a=np.random.randint(5,50)
        self.c_minus=np.random.randint(27,83)
        self.c_plus=np.random.randint(1,30)
        self.p_smj=np.random.rand(sim_setting['scenarios_max_index'],sim_setting['customer_max_index'],sim_setting["week_max_index"])*3
        self.r_iq = np.around(np.random.rand(sim_setting['crop_max_index'],sim_setting['disease_max_index']), 0)
        self.u_q=np.random.rand(sim_setting['disease_max_index'])*3


        logging.info(f"crops: {self.crops}")
        logging.info(f"varieties: {self.varieties}")
        logging.info(f"weeks: {self.weeks}")
        logging.info(f"bands: {self.bands}")
        logging.info(f"customers: {self.customers}")
        logging.info(f"diseases: {self.diseases}")
        logging.info(f"scenarios: {self.scenarios}")
        logging.info(f"prob s: {self.prob_s}")
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
        logging.info(f"sowingWeeks: {self.sowingWeeks}")
        logging.info(f"spacings: {self.spacings}")
        logging.info("simulation end")

    def get_data(self):
        logging.info("getting data from instance...")
        return {
            "crops": self.crops,
            "varieties": self.varieties,
            "weeks": self.weeks,
            "bands": self.bands,
            "customers": self.customers,
            "diseases": self.diseases,
            "scenarios": self.scenarios,
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
            "sowingWeeks": self.sowingWeeks,
            "spacings": self.spacings,
        }
