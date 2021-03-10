import itertools
import datetime
from multiprocessing import Pool, cpu_count
import os 
import sys

import numpy as np
import torch
import cvxpy as cp

from logic_learning import Logic_Learning_Model, Timer

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
        self.Time_tolerance = 0.1
        self.body_predicate_set = list() # the index set of all body predicates
        self.head_predicate_set = list() # the index set of all head predicates
        self.decay_rate = 1 # decay kernel
        self.predicate_notation = list()

        self.body_intensity= dict()
        self.logic_template = dict()
        self.model_parameter = dict()
        self.time_horizon = 0
        
    def get_model(self):
        model = Logic_Learning_Model(self.head_predicate_set)
        model.logic_template = self.logic_template
        model.model_parameter = self.model_parameter
        model.num_predicate = self.num_predicate
        model.num_formula = self.num_formula
        model.predicate_set = self.body_predicate_set + self.head_predicate_set 
        model.predicate_notation = self.predicate_notation
        model.Time_tolerance = self.Time_tolerance
        model.decay_rate = self.decay_rate
        return model
    
    def get_model_for_learn(self):
        # used in logic_learning.py
        model = Logic_Learning_Model(self.head_predicate_set)
        model.num_predicate = self.num_predicate
        model.predicate_set = self.body_predicate_set + self.head_predicate_set 
        model.predicate_notation = self.predicate_notation
        return model

    def generate_one_sample(self, sample_ID=0):
        data_sample = dict()
        for predicate_idx in range(self.num_predicate):
            data_sample[predicate_idx] = {}
            data_sample[predicate_idx]['time'] = []
            data_sample[predicate_idx]['state'] = []

        # generate data (body predicates)
        for body_predicate_idx in self.body_predicate_set:
            t = 0
            while t < self.time_horizon:
                time_to_event = np.random.exponential(scale=1.0 / self.body_intensity[body_predicate_idx])
                t += time_to_event
                if t >= self.time_horizon:
                    break
                data_sample[body_predicate_idx]['time'].append(t)
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
                    cur_state = 1 - data_sample[head_predicate_idx]['state'][-1]
                    data_sample[head_predicate_idx]['state'].append(cur_state)
        return data_sample

    def generate_data(self, num_sample, time_horizon, worker_num):
        self.model = self.get_model()
        self.time_horizon = time_horizon
        print("Generate {} samples".format(num_sample))
        print("with following rules:")
        self.model.print_rule_cp()
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
            model.model_parameter[head_predicate_idx] = dict()
            model.model_parameter[head_predicate_idx]['base'] = torch.autograd.Variable((torch.ones(1) * -0.2).double(), requires_grad=True)
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

    def fit_gt_rules_mp(self, dataset, time_horizon):
        # fit logic learning model
        model = self.get_model()
        model.num_iter = 100
        model.print_info()
        # initialize params
        for head_predicate_idx in self.head_predicate_set:
            model.model_parameter[head_predicate_idx] = dict()
            model.model_parameter[head_predicate_idx]['base'] = torch.autograd.Variable((torch.ones(1) * -0.2).double(), requires_grad=True)
            for formula_idx in range(self.num_formula):
                model.model_parameter[head_predicate_idx][formula_idx] = {'weight': torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)}
        verbose = True # print mid-result
        l = model.optimize_log_likelihood_mp(head_predicate_idx, dataset, time_horizon,verbose)
        print("Log-likelihood (torch)= ", l, flush=1)
        print("rule set is:")
        model.print_rule()


        

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
    model.model_parameter[head_predicate_idx] = {'base':torch.tensor([0]).double()}
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
    model.model_parameter[head_predicate_idx] = {'base':torch.tensor([0]).double()}
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

    model.model_parameter[head_predicate_idx] = {'base':torch.tensor([0]).double()}
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

    model.model_parameter[head_predicate_idx] = {'base':torch.tensor([0]).double()}
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
    # generate to data-5.npy
    file_name = "data-5.npy"
    
    model = Logic_Model_Generator()
    model.body_intensity= {0:1.0, 1:1.0, 2:1.0, 3:1.0}
    model.body_predicate_set = [0,1,2,3]
    model.head_predicate_set = [4]
    model.predicate_notation = ['A','B','C','D','E']
    model.num_predicate = len(model.body_predicate_set)
    
    from utils import get_template
    rule_set_str = """Head:E, base(torch)=0.1411, base(cp)=0.1411,
                        Rule0: A --> E , A BEFORE E, weight(torch)=0.7962, weight(cp)=0.7962.
                        Rule1: Not C --> E , Not C BEFORE E, weight(torch)=0.5391, weight(cp)=0.5391.
                        Rule2: D --> E , D EQUAL E, weight(torch)=0.4747, weight(cp)=0.4747.
                        Rule3: Not A --> E , Not A EQUAL E, weight(torch)=0.5983, weight(cp)=0.5983.
                        Rule4: Not C --> Not E , Not C EQUAL Not E, weight(torch)=0.3989, weight(cp)=0.3989.
                        Rule5: Not B ^ A --> E , Not B EQUAL E ^ A BEFORE E, weight(torch)=1.3190, weight(cp)=1.3190.
                        Rule6: D ^ Not A --> E , D EQUAL Not A ^ Not A EQUAL E, weight(torch)=0.01, weight(cp)=0.01.
                        Rule7: D ^ Not A --> E , D EQUAL E ^ Not A EQUAL E, weight(torch)=0.01, weight(cp)=0.01."""
    model_parameter, logic_template, head_predicate_idx, num_formula = get_template(rule_set_str, model.predicate_notation)
    model.logic_template = logic_template
    model.model_parameter = model_parameter
    model.head_predicate_set = [head_predicate_idx]
    model.num_formula = num_formula
    
    return model, file_name

def generate(model,file_name, num_sample, time_horizon, worker_num):
    with Timer("Generate data") as t:
        data = model.generate_data(num_sample=num_sample, time_horizon=time_horizon, worker_num=worker_num)
    if not os.path.exists("./data"):
        os.makedirs("./data")
    path = os.path.join("./data", file_name)
    np.save(path, data)
    print("data saved to", path)

def load_data(file_name):
    path = os.path.join("./data", file_name)
    data = np.load(path, allow_pickle='TRUE').item()
    return data

def fit(model,file_name,time_horizon):
    data = load_data(file_name)
    with Timer("Fit data") as t:
        model.fit_gt_rules(data, time_horizon=time_horizon)

def fit_mp(model,file_name,time_horizon):
    data = load_data(file_name)
    with Timer("Fit data (torch)") as t:
        model.fit_gt_rules_mp(data, time_horizon=time_horizon)

def fit_cp(model,file_name,time_horizon):
    data = load_data(file_name)
    with Timer("Fit data (cp)") as t:
        model.fit_gt_rules_cp(data, time_horizon=time_horizon)

def avg_event_num(file_name):
    data = load_data(file_name)
    event_num_dict = {pred_idx:0 for pred_idx in data[0].keys()}
    for sample_ID in data.keys():
        for pred_idx in event_num_dict.keys():
            event_num_dict[pred_idx] += len(data[sample_ID][pred_idx]['time'])
    
    num_sample = len(data.keys())
    print("data file is ", file_name)
    print("num_sample = {}".format(num_sample))
    for pred_idx, cnt in event_num_dict.items():
        print("pred {} avg event num is {:.4f}".format(pred_idx, cnt/num_sample))
    
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

def get_model_by_idx(model_idx):
    model_list = [None, get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5]
    return model_list[model_idx]()

if __name__ == "__main__":
    #redirect_log_file()
    print("Start time is", datetime.datetime.now(),flush=1)
    num_sample = 64
    time_horizon = 10
    worker_num = 24
    model_idx = 4
    model, file_name = get_model_by_idx(model_idx = model_idx)

    #generate(model, file_name, num_sample, time_horizon, worker_num)
    print("generate finish",flush=1)
    avg_event_num(file_name)
    feature_mean_std(model_idx = model_idx)
    fit_cp(model, file_name, time_horizon)
    
