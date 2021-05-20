# -*- coding: utf-8 -*-
import logging
import numpy as np


class Instance():
    def __init__(self, sim_setting):
        logging.info("starting simulation...")
        self.crops = sim_setting['crop_max_index']
        self.varieties = sim_setting['variety_max_index']
        self.weeks = sim_setting['week_max_index']
        self.bands = sim_setting['size_band_max_index']
        self.customers = sim_setting['customer_max_index']
        self.diseases = sim_setting['disease_max_index']
        self.scenarios = sim_setting['scenarios_max_index']

        self.prob_s = np.around(np.random.uniform(0,1,sim_setting['scenarios_max_index']), 2)
        self.prob_s = self.prob_s/np.sum(self.prob_s)


        self.crops_ss = sim_setting['crop_max_index']

        logging.info(f"crops: {self.crops}")
        logging.info(f"varieties: {self.varieties}")
        logging.info(f"weeks: {self.weeks}")
        logging.info(f"bands: {self.bands}")
        logging.info(f"customers: {self.customers}")
        logging.info(f"diseases: {self.diseases}")
        logging.info(f"scenarios: {self.scenarios}")
        logging.info(f"prob s: {self.prob_s}")
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
        }
