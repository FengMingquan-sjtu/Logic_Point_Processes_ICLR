import itertools
import datetime
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, cpu_count
import os 
import sys
import pickle 

import numpy as np
import torch
#import cvxpy as cp

from logic_learning import Logic_Learning_Model
from utils import Timer, get_data

##################################################################
#np.random.seed(1)
class Logic_Model_Generator:
    def __init__(self):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 0  # num_predicate is same as num_node
        self.num_formula = 0
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.STATIC = "STATIC"
        self.Time_tolerance = 0.1
        self.body_predicate_set = list() # the index set of all body predicates
        self.head_predicate_set = list() # the index set of all head predicates
        self.static_pred_set = list()
        self.instant_pred_set = list()
        self.decay_rate = 1 # decay kernel
        self.predicate_notation = list()

        self.body_intensity= dict()
        self.logic_template = dict()
        self.model_parameter = dict()
        self.time_horizon = 0
        self.integral_resolution = 0.1
        self.use_2_bases = False
        self.use_exp_kernel = False
        
        
        
    def get_model(self):
        # used in fit-gt.
        model = Logic_Learning_Model(self.head_predicate_set)
        model.logic_template = self.logic_template
        model.model_parameter = self.model_parameter
        model.num_predicate = self.num_predicate
        model.num_formula = self.num_formula
        model.predicate_set = self.body_predicate_set + self.head_predicate_set 
        model.static_pred_set = self.static_pred_set
        model.instant_pred_set = self.instant_pred_set
        model.predicate_notation = self.predicate_notation
        model.Time_tolerance = self.Time_tolerance
        model.decay_rate = self.decay_rate
        model.integral_resolution = self.integral_resolution
        model.use_2_bases = self.use_2_bases
        model.use_exp_kernel = self.use_exp_kernel
        
        return model
    
    def get_model_for_learn(self):
        # used in logic_learning.py\ train_synthetic.py
        model = Logic_Learning_Model(self.head_predicate_set)
        model.num_predicate = self.num_predicate
        model.predicate_set = self.body_predicate_set + self.head_predicate_set 
        model.predicate_notation = self.predicate_notation
        model.static_pred_set = self.static_pred_set
        model.instant_pred_set = self.instant_pred_set
        model.body_pred_set = self.body_predicate_set
        model.predicate_notation = self.predicate_notation
        model.Time_tolerance = self.Time_tolerance
        model.decay_rate = self.decay_rate
        model.integral_resolution = self.integral_resolution
        model.use_2_bases = self.use_2_bases
        model.use_exp_kernel = self.use_exp_kernel
        model.init_params()
        return model

    def generate_one_sample(self, sample_ID=0):
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
            #print(intensity_max)
            # generate events via accept and reject
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
        self.model = self.get_model()
        self.time_horizon = time_horizon
        print("Generate {} samples".format(num_sample))
        print("with following rules:")
        self.model.print_rule_cp()
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

    def fit_gt_rules(self, dataset, time_horizon):
        # fit logic learning model
        model = self.get_model()
        model.num_iter = 100
        model.print_info()
        # initialize params
        for head_predicate_idx in self.head_predicate_set:
            for formula_idx in range(self.num_formula):
                model.model_parameter[head_predicate_idx][formula_idx] = {'weight': torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)}
        verbose = True # print mid-result
        l = model.optimize_log_likelihood(head_predicate_idx, dataset, time_horizon,verbose)
        print("Log-likelihood (torch)= ", l, flush=1)
        print("rule set is:")
        model.print_rule()
    
    def fit_gt_rules_cp(self, dataset, time_horizon):
        model = self.get_model()
        model.print_info()

        # initialize params
        for head_predicate_idx in self.head_predicate_set:
            model.model_parameter[head_predicate_idx] = dict()
            model.model_parameter[head_predicate_idx]['base_cp'] = cp.Variable(1)
            for formula_idx in range(self.num_formula):
                model.model_parameter[head_predicate_idx][formula_idx] = {'weight_cp': cp.Variable(1)}
        
        l = model.optimize_log_likelihood_cp(head_predicate_idx, dataset, time_horizon)
        print("Log-likelihood (cp)= ", l, flush=1)
        print("rule set is:")
        model.print_rule_cp()

    def fit_gt_rules_mp(self, dataset, time_horizon, num_iter, worker_num):
        # fit logic learning model
        model = self.get_model()
        model.num_iter = num_iter
        model.worker_num = worker_num
        model.print_info()
        # initialize params
        init_base = 0.2
        init_weight = 0.1
        for head_predicate_idx in self.head_predicate_set:
            model.model_parameter[head_predicate_idx] = {'base': torch.autograd.Variable((torch.ones(1) * init_base).double(), requires_grad=True)}
            for formula_idx in range(self.num_formula):
                model.model_parameter[head_predicate_idx][formula_idx] = {'weight': torch.autograd.Variable((torch.ones(1) * init_weight).double(), requires_grad=True)}
        verbose = True # print mid-result
        l = model.optimize_log_likelihood_mp(head_predicate_idx, dataset,verbose)
        print("Log-likelihood (torch)= ", l, flush=1)
        print("rule set is:")
        model.print_rule()
        return model


        

def get_logic_model_1():
    # generate to data-1.npy
    file_name = "data-1.npy"

    model = Logic_Model_Generator()
    model.body_intensity= {0:2, 1:2, 2:2, 3:2}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4
    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [0.3, 0.4, 0.5, 0.6, 0.7]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}

    head_predicate_idx = 4
    logic_template[head_predicate_idx] = {} 

    # A ^ C ->E  A Before E, C Before E.
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 4], [2,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # A ^ B --> E,  A Before E, B Equal E.
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0,1]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,4], [1,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE,model.EQUAL]

    # B ^ C ^ D --> E, B Equal E, C Before E, D Before E.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,4], [2,4], [3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL, model.BEFORE, model.BEFORE]

    # C ^ D --> E,  C Equal E, D BEFORE E.
    formula_idx = 3
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2, 4],[3,4]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL, model.BEFORE]

    # Not C --> E, Not C Before E
    formula_idx = 4
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [0]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,4] ]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_2():
    # generate to data-2.npy
    file_name = "data-2.npy"
    # E is noise variable
    model = Logic_Model_Generator()
    model.body_intensity= {0:2, 1:2, 2:2, 3:2, 4:2}
    model.body_predicate_set = [0,1,2,3,4]
    model.head_predicate_set = [5]
    model.predicate_notation = ['A','B','C','D','E','F']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 5
    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [0.8, 0.9, 0.5, 0.6, 0.7]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A ^ C ->F  A Before F, C Before F.
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 5], [2, 5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # A ^ B ^ C --> F,  A Before F, B Equal F, C Before F.
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0,1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,5], [1,5], [2,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL, model.BEFORE]

    # B ^ C ^ D --> F, B Equal F, C Before F, D Before F.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,5], [2,5], [3,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL, model.BEFORE, model.BEFORE]

    # C ^ D --> F,  C Equal F, D BEFORE F.
    formula_idx = 3
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2, 5],[3,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL, model.BEFORE]

    # Not C --> F, Not C Before F
    formula_idx = 4
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [0]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,5] ]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_3():
    # generate to data-3.npy
    file_name = "data-3.npy"
    # E,F are noise variables
    model = Logic_Model_Generator()
    model.body_intensity= {0:2.1, 1:2.2, 2:1.9, 3:1.8, 4:2.3, 5:1.7}
    model.body_predicate_set = [0,1,2,3,4,5]
    model.head_predicate_set = [6]
    model.predicate_notation = ['A','B','C','D','E','F','G']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 6

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([0]).double()}
    weights = [0.9, 0.6, 1.0, 0.8, 0.5]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A ^ C ->G  A Before G, C Before G.
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 6], [2, 6]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE]

    # A ^ B ^ C --> G,  A Before F, B Equal F, C Before G.
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0,1,2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,6], [1,6], [2,6]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.EQUAL, model.BEFORE]

    # B ^ C ^ D --> G, B Equal G, C Before G, D Before G.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1,2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,6], [2,6], [3,6]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL, model.BEFORE, model.BEFORE]

    # C ^ D --> G,  C Equal G, D BEFORE G.
    formula_idx = 3
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2, 6],[3,6]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL, model.BEFORE]

    # Not C --> G, Not C Before G
    formula_idx = 4
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [2]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [0]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[2,6] ]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE]

    model.logic_template = logic_template
    
    return model, file_name

def get_logic_model_4():
    # generate to data-4.npy
    file_name = "data-4.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
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

def get_logic_model_5():
    # only difference with model_4 is weights.
    # generate to data-5.npy
    file_name = "data-5.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
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

def get_logic_model_6():
    # only difference with model_4 is weights.
    
    file_name = "data-6.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
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

def get_logic_model_7():
    # difference with model_4 is rule length. data-4 is [1,2,2], data-7 is [1,2,3]
    
    file_name = "data-7.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
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

def get_logic_model_8():
    # difference with model_4 is one more rule.
   
    file_name = "data-8.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0 }
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
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

def get_logic_model_9():
    # difference with model_4 is one more predicate.
    # E is dummy pred, F is target 
    
    file_name = "data-9.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0, 4:0.2} #dummy E has lower intensity
    model.body_predicate_set = [0,1,2,3,4]
    model.head_predicate_set = [5]
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

def get_logic_model_10():
    '''!!! Deprecated!!! '''
    #difference with data-4 is seperate bases for 0->1 and 1->0
    file_name = "data-10.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4

    model.model_parameter[head_predicate_idx] = {'base_0_1':torch.tensor([0.1]).double(), 'base_1_0':torch.tensor([-0.1]).double()}
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

def get_logic_model_11():
    # difference with data-4: various body intensity.
    file_name = "data-11.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
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

def get_logic_model_12():
    # only difference with model_11 is smaller weights.
    # modified from data-5.npy
    file_name = "data-12.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
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

def get_logic_model_13():
    # only difference with model_11 is large weights.
    # modified from data-6
    
    file_name = "data-13.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
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

def get_logic_model_14():
    # difference with model_11 is rule length. data-11 is [1,2,2], data-14 is [1,2,3]
    # modified from data-7
    file_name = "data-14.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.instant_pred_set = [4]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4
    init_base = 0.0

    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([init_base]).double()}
    weights = [-1.0, -1.0, -2.0]
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

def get_logic_model_15():
    # difference with model_11 is one more rule.
    #modiied from data-8
    file_name = "data-15.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 4
    init_base = 0.5

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

def get_logic_model_16():
    # difference with model_11 is one dummy predicate.
    # E is dummy pred, F is target 
    
    # modified from data-9
    file_name = "data-16.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:1.4, 4:0.2} #dummy E has lower intensity
    model.body_predicate_set = [0,1,2,3,4]
    model.head_predicate_set = [5]
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

def get_logic_model_17():
    # test self-exciting and static variable
    # E is dummy pred, F is target 
    
    # modified from data-16
    file_name = "data-17.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.6, 1:0.8, 2:1.2, 3:0.1, 4:0.1}  #D and E are static
    model.body_predicate_set = [0,1,2,3,4]
    model.static_pred_set = [3,4]
    model.head_predicate_set = [5]
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

    # F ^ D --> F, F Before F, D Static F
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [5, 3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1, 1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[5, 5], [3, 5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.STATIC]

    # A ^ D --> F,  A Before F, D Static F
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0,3]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,5], [3,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.STATIC]

    # B ^ C ^ E --> F, B Before F, C Before F, E Static F.
    formula_idx = 2
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1, 2, 4]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1,1,1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,5], [2,5], [4,5]]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.BEFORE, model.BEFORE, model.STATIC]

    model.logic_template = logic_template
    
    return model, file_name


def get_logic_model_18():
    # test self-exciting 
    # E is dummy pred, F is target 
    
    # modified from data-17
    file_name = "data-18.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:0.1}  
    model.body_predicate_set = [0]
    model.static_pred_set = []
    model.instant_pred_set = [0, 1]
    model.head_predicate_set = [1]
    model.predicate_notation = ['A', 'B']
    model.num_predicate = len(model.body_predicate_set)
    model.Time_tolerance = 12
    model.decay_rate = 0.01
    model.time_window = 12
    model.integral_resolution = 0.5
    
    # define weights and base
    model.model_parameter = dict()
    head_predicate_idx = 1
    base = -5
    model.model_parameter[head_predicate_idx] = { 'base':torch.tensor([base,]).double()}
    weights = [1, 1]
    model.num_formula = len(weights)
    for idx, w in enumerate(weights):
        model.model_parameter[head_predicate_idx][idx] = {'weight': torch.tensor([w]).double()}
   
    # encode rule information
    logic_template = {}
    logic_template[head_predicate_idx] = {} 

    # A --> B, A Equal B
    formula_idx = 0
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0,1],]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL]

    # B --> B, B Equal B
    formula_idx = 1
    logic_template[head_predicate_idx][formula_idx] = {}
    logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [1]
    logic_template[head_predicate_idx][formula_idx]['body_predicate_sign'] = [1]  # use 1 to indicate True; use 0 to indicate False
    logic_template[head_predicate_idx][formula_idx]['head_predicate_sign'] = [1]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[1,1],]
    logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [model.EQUAL]

    model.logic_template = logic_template
    
    return model, file_name

def get_model_by_idx(model_idx):
    model_list = [None, get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5,get_logic_model_6,get_logic_model_7,get_logic_model_8,get_logic_model_9,get_logic_model_10,get_logic_model_11,get_logic_model_12,get_logic_model_13,get_logic_model_14,get_logic_model_15,get_logic_model_16, get_logic_model_17, get_logic_model_18]
    return model_list[model_idx]()



def load_data(file_name, num_sample=0):
    path = os.path.join("./data", file_name)
    data = np.load(path, allow_pickle='TRUE').item()
    if num_sample > 0:
        num_sample = min(num_sample, len(data.keys()))
        data = {i:data[i] for i in range(num_sample)}
    return data

def avg_event_num(model_idx, num_sample):
    model, file_name = get_model_by_idx(model_idx)
    data = load_data(file_name, num_sample)
    event_num_dict = {pred_idx:0 for pred_idx in data[0].keys()}
    for sample_ID in data.keys():
        for pred_idx in event_num_dict.keys():
            event_num_dict[pred_idx] += len(data[sample_ID][pred_idx]['time'])
    
    num_sample = len(data.keys())
    print("data file is ", file_name)
    print("num_sample = {}".format(num_sample))
    for pred_idx, cnt in event_num_dict.items():
        print("pred {} avg event num is {:.4f}".format(pred_idx, cnt/num_sample))
    

def fit_mp(model_idx, num_sample, time_horizon, num_iter, worker_num ):
    print("---- start  fit_mp ----")
    model, file_name = get_model_by_idx(model_idx)
    data = load_data(file_name, num_sample)
    avg_event_num(model_idx, num_sample)
    print("fit data-{}, with {} samples".format(model_idx, num_sample))
    with Timer("Fit data (torch)") as t:
        m = model.fit_gt_rules_mp(data, time_horizon, num_iter, worker_num)
    tag = "fit-gt-{}".format(model_idx)
    with open("./model/model-{}.pkl".format(tag),'wb') as f:
        pickle.dump(m, f)
    print("---- exit  fit_mp ----")

def fit_cp(model_idx, num_sample, time_horizon ):
    model, file_name = get_model_by_idx(model_idx)
    data = load_data(file_name, num_sample)
    with Timer("Fit data (cp)") as t:
        model.fit_gt_rules_cp(data, time_horizon=time_horizon)

def generate(model_idx, num_sample, time_horizon, worker_num):
    print("---- start  generate ----")
    model, file_name = get_model_by_idx(model_idx)
    with Timer("Generate data") as t:
        data = model.generate_data(num_sample=num_sample, time_horizon=time_horizon, worker_num=worker_num)
    if not os.path.exists("./data"):
        os.makedirs("./data")
    path = os.path.join("./data", file_name)
    np.save(path, data)
    avg_event_num(model_idx, num_sample)
    print("data saved to", path)
    print("---- exit  generate ----")

def feature_mean_std(model_idx):
    model, file_name = get_model_by_idx(model_idx = model_idx)
    head_predicate_idx = model.head_predicate_set[0]
    learning_model = model.get_model()
    data = load_data(file_name)
    ret_dict = dict()
    for sample_ID, history in data.items():
        for formula_idx, template in learning_model.logic_template[head_predicate_idx].items():
            for cur_time in range(1,10):
                f = learning_model.get_feature(cur_time, head_predicate_idx, history, template)
                if not (formula_idx,cur_time) in ret_dict:
                    ret_dict[(formula_idx,cur_time)] = list()
                ret_dict[(formula_idx,cur_time)].append(f.item()) 

    print("featrue mean and std.")
    print("(f, t) --> f is formula_idx, t is time.")
    for k in ret_dict.keys():
        mean = np.mean(ret_dict[k])
        std = np.std(ret_dict[k], ddof=1)
        print(k, "mean={:.4f}, std={:.4f}".format(mean, std))
    


def redirect_log_file():
    log_root = ["./log/out","./log/err"]   
    for root in log_root:     
        if not os.path.exists(root):
            os.makedirs(root)
    t = str(datetime.datetime.now())
    out_file = os.path.join(log_root[0], t)
    err_file = os.path.join(log_root[1], t)
    sys.stdout = open(out_file, 'w')
    sys.stderr = open(err_file, 'w')

def fit_mp_group(model_idx):
    fit_mp(model_idx=model_idx, num_sample=600, time_horizon=10, num_iter = 50, worker_num = 12 )
    fit_mp(model_idx=model_idx, num_sample=1200, time_horizon=10, num_iter = 50, worker_num = 12 )
    fit_mp(model_idx=model_idx, num_sample=2400, time_horizon=10, num_iter = 50, worker_num = 12 )


def test(dataset_name, num_sample, model_file, head_predicate_idx, worker_num):
    dataset,num_sample =  get_data(dataset_name=dataset_name, num_sample=num_sample)
    testing_dataset = {i: dataset[int(num_sample*0.8)+i] for i in range(int(num_sample*0.2))}
    with open("./model/"+model_file, "rb") as f:
        model = pickle.load(f)
    model.worker_num = worker_num
    model.generate_target(head_predicate_idx=head_predicate_idx, dataset=testing_dataset, num_repeat=100)

if __name__ == "__main__":
    redirect_log_file()
    torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78

    print("Start time is", datetime.datetime.now(),flush=1)
    
    #test(dataset_name="data-18", num_sample=-1, model_file="model-fit-gt-18.pkl", head_predicate_idx=1, worker_num=12)
    #generate(model_idx=8, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=8)

    #generate(model_idx=9, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=9)

    #generate(model_idx=10, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=10)

    #generate(model_idx=11, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=11)

    #generate(model_idx=4, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=4)

    #generate(model_idx=12, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=12)

    #generate(model_idx=13, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=13)

    #generate(model_idx=14, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=14)

    generate(model_idx=15, num_sample=2400, time_horizon=10, worker_num=12)
    fit_mp_group(model_idx=15)

    #generate(model_idx=16, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=16)
    #generate(model_idx=17, num_sample=2400, time_horizon=10, worker_num=12)
    #fit_mp_group(model_idx=17)

    #generate(model_idx=18, num_sample=2400, time_horizon=168, worker_num=12)
    #fit_mp(model_idx=18, num_sample=1200, time_horizon=168, num_iter = 12, worker_num = 12 )

    #data = load_data(file_name="data-18.npy")
    #print(data[0])
    