import itertools
import datetime
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, cpu_count
import os 
import sys
import pickle 
import argparse

import numpy as np
import torch

from logic_learning import Logic_Learning_Model
from utils import Timer, redirect_log_file

class Logic_Model_Generator:
    def __init__(self):
        self.num_predicate = 0  
        self.num_formula = 0
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.Time_tolerance = 0.1
        self.body_predicate_set = list() # the index set of all body predicates
        self.head_predicate_set = list() # the index set of all head predicates
        self.instant_pred_set = list()
        self.decay_rate = 1 # decay kernel
        self.predicate_notation = list()

        self.body_intensity= dict()
        self.logic_template = dict()
        self.model_parameter = dict()
        self.time_horizon = 0
        self.integral_resolution = 0.1
        self.use_2_bases = False
        self.use_exp_kernel = True
        self.reverse_head_sign = True

    def get_model(self):
        # get model for generate
        model = Logic_Learning_Model(self.head_predicate_set)
        model.logic_template = self.logic_template
        model.model_parameter = self.model_parameter
        model.num_predicate = self.num_predicate
        model.num_formula = self.num_formula
        model.predicate_set = self.body_predicate_set + self.head_predicate_set 
        model.instant_pred_set = self.instant_pred_set
        model.predicate_notation = self.predicate_notation
        model.Time_tolerance = self.Time_tolerance
        model.decay_rate = self.decay_rate
        model.integral_resolution = self.integral_resolution
        model.use_2_bases = self.use_2_bases
        model.use_exp_kernel = self.use_exp_kernel
        model.reverse_head_sign = self.reverse_head_sign
        
        return model

    def get_model_for_learn(self):
        #get the model used in train_synthetic.py
        model = Logic_Learning_Model(self.head_predicate_set)
        model.num_predicate = self.num_predicate
        model.predicate_set = self.body_predicate_set + self.head_predicate_set 
        model.predicate_notation = self.predicate_notation
        model.instant_pred_set = self.instant_pred_set
        model.body_pred_set = self.body_predicate_set
        model.predicate_notation = self.predicate_notation
        model.Time_tolerance = self.Time_tolerance
        model.decay_rate = self.decay_rate
        model.integral_resolution = self.integral_resolution
        model.use_2_bases = self.use_2_bases
        model.use_exp_kernel = self.use_exp_kernel
        model.reverse_head_sign = self.reverse_head_sign
        model.init_params()
        return model

    def generate_one_sample(self, sample_ID=0):
        """generate a point process sample, guided by logic rule, via Ogata Thinning algorithm.
        Input
        --------
        sample_ID: an interger index for each sample.

        Output:
        --------
        data_sample: a nested dict, data_sample[predicate_idx]["time"] is a list of occurance time of the predicate_idx
        and data_sample[predicate_idx]["state"] is the 0-1 state list of it.
        """
        data_sample = dict()
        for predicate_idx in range(self.num_predicate):
            data_sample[predicate_idx] = {}
            data_sample[predicate_idx]['time'] = [0,]
            data_sample[predicate_idx]['state'] = [1,]

        # generate data (body predicates)
        for body_predicate_idx in self.body_predicate_set:
            t = 0
            while t < self.time_horizon:
                time_to_event = np.random.exponential(scale=1.0 / self.body_intensity[body_predicate_idx])
                t += time_to_event
                if t >= self.time_horizon:
                    break
                data_sample[body_predicate_idx]['time'].append(t)
                if body_predicate_idx in self.instant_pred_set:
                    cur_state = 1
                else:
                    if len(data_sample[body_predicate_idx]['state'])>0:
                        cur_state = 1 - data_sample[body_predicate_idx]['state'][-1]
                    else:
                        cur_state = 1
                data_sample[body_predicate_idx]['state'].append(cur_state)

        # generate head predicate data.
        for head_predicate_idx in self.head_predicate_set:
            data_sample[head_predicate_idx] = {}
            data_sample[head_predicate_idx]['time'] = [0,]
            data_sample[head_predicate_idx]['state'] = [0,]

            data = {sample_ID:data_sample}
            # obtain the maximal intensity
            intensity_potential = []
            
            for t in np.arange(0, self.time_horizon, 0.1):
                t = t.item() #convert np scalar to float
                intensity = self.model.intensity(t, head_predicate_idx, data, sample_ID)
                intensity_potential.append(intensity)
            intensity_max = max(intensity_potential)
            
            # generate events via accept and reject (thinning)
            t = 0
            while t < self.time_horizon:
                time_to_event = np.random.exponential(scale=1.0/intensity_max).item()
                t = t + time_to_event
                if t >= self.time_horizon:
                    break
                intensity = self.model.intensity(t, head_predicate_idx, data, sample_ID)
                ratio = min(intensity / intensity_max, 1)
                flag = np.random.binomial(1, ratio)     # if flag = 1, accept, if flag = 0, regenerate
                if flag == 1: # accept
                    data_sample[head_predicate_idx]['time'].append(t)
                    if head_predicate_idx in self.instant_pred_set:
                        cur_state = 1
                    else:
                        cur_state = 1 - data_sample[head_predicate_idx]['state'][-1]
                    data_sample[head_predicate_idx]['state'].append(cur_state)
        return data_sample

    def generate_data(self, num_sample, time_horizon, worker_num):
        """generate a synthetic dataset in parallel.
        
        Input
        -------
        num_sample: the number of samples in dataset
        time_horizon: the time span of each sample
        worker_num: the number of parallel cores.

        Output 
        _______
        data: a nested dict of data samples, key: Sample_ID, value: sample.
        """
        self.model = self.get_model()
        self.time_horizon = time_horizon
        print("Generate {} samples".format(num_sample))
        print("with following rules:")
        self.model.print_rule()
        print("with following settings:")
        self.model.print_info()
        for body_idx in self.body_predicate_set:
            print("Intensity {} is {}".format(self.predicate_notation[body_idx], self.body_intensity[body_idx]))
        print("-----",flush=1)

        
        cpu = cpu_count()
        worker_num_ = min(worker_num, cpu)
        print("cpu num = {}, use {} workers. ".format(cpu, worker_num_))
        
        if worker_num_ > 1:
            with torch.no_grad():
                with Pool(worker_num_) as pool:
                    samples = pool.map(self.generate_one_sample, range(num_sample))
        else:
            samples = [self.generate_one_sample() for i in range(num_sample)]
        
        data = dict(enumerate(samples))
            
        return data



        
"""
The following functions, from get_logic_model_1 to get_logic_model_12, define 12 logic rule sets.
"""

def get_logic_model_1():
    file_name = "data-1.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [1.0, 1.0, 1.0]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_2():
    # only difference with model_1 is weights.
    file_name = "data-2.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [0.5, 0.5, 0.5]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_3():
    # only difference with model_1 is weights.
    
    file_name = "data-3.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [1.5, 1.5, 1.5]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_4():
    # difference with model_1 is rule length. data-4 is [1,2,2], data-7 is [1,2,3]
    
    file_name = "data-4.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [1.0, 1.0, 2.0]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # A ^ C ^ D --> E, A Before E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0, 2, 3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1, 1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,4], [2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_5():
    # difference with model_1 is one more rule.
   
    file_name = "data-5.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0 }
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [1.0, 1.0, 1.0, 1.0]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2, 3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    # A ^ B --> E, A Before E, B Before E.
    formula_idx = 3
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0, 1]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,4], [1,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_6():
    # difference with model_1 is one more predicate.
    # E is dummy pred, F is target 
    
    file_name = "data-6.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0, 4:0.2} #dummy E has lower intensity
    model.body_predicate_set = [0,1,2,3,4]
    model.head_predicate_set = [5]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E','F']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 5

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [1.0, 1.0, 1.0]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> F, A Before F
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> F,  B Before F, C Before F
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,5], [2,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> F, C Before F, D Equal F.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2, 3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,5], [3,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_7():
    # difference with data-1: various body intensity.
    file_name = "data-7.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [1.0, 1.0, 1.0]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_8():
    # only difference with model_7 is smaller weights.
    file_name = "data-8.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [0.5, 0.5, 0.5]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_9():
    # only difference with model_7 is large weights.
    
    file_name = "data-9.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [1.5, 1.5, 1.5]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_10():
    # difference with model_7 is rule length. data-11 is [1,2,2], data-14 is [1,2,3]
    file_name = "data-10.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4
    init_base = 0.0

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([init_base]).double()}
    weights = [1.0, 1.0, 2.0]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # A ^ C ^ D --> E, A Before E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0, 2, 3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1, 1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,4], [2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_11():
    # difference with model_7 is one more rule.
    file_name = "data-11.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4
    init_base = 0.0

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([init_base]).double()}
    weights = [1.0, 1.0, 1.0, 1.0]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> E, A Before E
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> E,  B Before E, C Before E
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> E, C Before E, D Equal E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2, 3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    # A ^ B --> E, A Before E, B Before E.
    formula_idx = 3
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0, 1]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,4], [1,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_12():
    # difference with model_7 is one dummy predicate.
    # E is dummy pred, F is target 
    file_name = "data-12.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4, 4:0.2} #dummy E has lower intensity
    model.body_predicate_set = [0,1,2,3,4]
    model.head_predicate_set = [5]
    model.instant_pred_set = [model.head_predicate_set[0]]
    model.predicate_notation = ['A','B','C','D','E','F']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 5

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [1.0, 1.0, 1.0]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> F, A Before F
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    # B ^ C --> F,  B Before F, C Before F
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,5], [2,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # C ^ D --> F, C Before F, D Equal F.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2, 3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,5], [3,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name


def get_model_by_idx(dataset_id):
    model_list = [None, get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5,get_logic_model_6,get_logic_model_7,get_logic_model_8,get_logic_model_9,get_logic_model_10,get_logic_model_11,get_logic_model_12]
    return model_list[dataset_id]()


def generate(dataset_id, num_sample, time_horizon, worker_num):
    """ The interface of generate data.
    dataset_id is the index of get_logic_model_x() functions, indicating which dataset to generate.
    """
    print("---- start  generate ----")
    model, file_name = get_model_by_idx(dataset_id)
    with Timer("Generate data") as t:
        data = model.generate_data(num_sample=num_sample, time_horizon=time_horizon, worker_num=worker_num)
    if not os.path.exists("./data"):
        os.makedirs("./data")
    path = os.path.join("./data", file_name)
    np.save(path, data)
    print("data saved to", path)
    print("---- exit  generate ----")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, 
        help = "an integer between 1 and 12, indicating one of 12 datasets",
        default = 1,
        choices = list(range(1,13)))
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    agrs = get_args()
    redirect_log_file() #redirect stdout and stderr to log files.
    torch.multiprocessing.set_sharing_strategy('file_system') #multi process communication strategy, depending on operating system.
    print("Start time is", datetime.datetime.now(),flush=1)

    generate(dataset_id=agrs.dataset_id, num_sample=2400, time_horizon=10, worker_num=12)
    
    