import itertools
import datetime

import numpy as np
import torch

from logic_learning_2 import Logic_Learning_Model, Timer

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

    def generate_data(self, num_sample, time_horizon):
        model = self.get_model()
        print("Generate {} samples".format(num_sample))
        print("with following rules:")
        model.print_rule()
        for body_idx in self.body_predicate_set:
            print("Intensity {} is {}".format(self.predicate_notation[body_idx], self.body_intensity[body_idx]))
        print("-----",flush=1)

        data={}

        for sample_ID in np.arange(0, num_sample, 1):
            data[sample_ID] = {}
            # initialize data
            for predicate_idx in np.arange(0, self.num_predicate, 1):
                data[sample_ID][predicate_idx] = {}
                data[sample_ID][predicate_idx]['time'] = []
                data[sample_ID][predicate_idx]['state'] = []

            # generate data (body predicates)
            for body_predicate_idx in self.body_predicate_set:
                t = 0
                while t < time_horizon:
                    time_to_event = np.random.exponential(scale=1.0 / self.body_intensity[body_predicate_idx])
                    t += time_to_event
                    if t >= time_horizon:
                        break
                    data[sample_ID][body_predicate_idx]['time'].append(t)
                    if len(data[sample_ID][body_predicate_idx]['state'])>0:
                        cur_state = 1 - data[sample_ID][body_predicate_idx]['state'][-1]
                    else:
                        cur_state = 1
                    data[sample_ID][body_predicate_idx]['state'].append(cur_state)


            for head_predicate_idx in self.head_predicate_set:
                data[sample_ID][head_predicate_idx] = {}
                data[sample_ID][head_predicate_idx]['time'] = [0,]
                data[sample_ID][head_predicate_idx]['state'] = [0,]

                # obtain the maximal intensity
                intensity_potential = []
                use_cache = False # do not need cache in generator.
                for t in np.arange(0, time_horizon, 0.1):
                    t = t.item() #convert np scalar to float
                    intensity_potential.append(model.intensity(t, head_predicate_idx, data, sample_ID, use_cache))
                intensity_max = max(intensity_potential)
                #print(intensity_max)
                # generate events via accept and reject
                t = 0
                while t < time_horizon:
                    time_to_event = np.random.exponential(scale=1.0/intensity_max).item()
                    t = t + time_to_event
                    if t >= time_horizon:
                        break
                    ratio = min(model.intensity(t, head_predicate_idx,  data, sample_ID, use_cache) / intensity_max, 1)
                    flag = np.random.binomial(1, ratio)     # if flag = 1, accept, if flag = 0, regenerate
                    if flag == 1: # accept
                        data[sample_ID][head_predicate_idx]['time'].append(t)
                        cur_state = 1 - data[sample_ID][head_predicate_idx]['state'][-1]
                        data[sample_ID][head_predicate_idx]['state'].append(cur_state)
                

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


        

def get_logic_model_1():
    # generate to data-1.npy
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
    
    return model

def get_logic_model_2():
    # generate to data-2.npy
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
    
    return model


def generate():
    logic_model_generator = get_logic_model_2()
    with Timer("Generate data") as t:
        data = logic_model_generator.generate_data(num_sample=1000, time_horizon=10)
        #print(data)
    np.save('data-2.npy', data)

def fit():
    logic_model_generator = get_logic_model_2()
    data = np.load('data-2.npy', allow_pickle='TRUE').item()
    with Timer("Fit data") as t:
        logic_model_generator.fit_gt_rules(data, time_horizon=10)

if __name__ == "__main__":
    print("Start time is", datetime.datetime.now(),flush=1)
    generate()
    print("generate finish",flush=1)
    fit()
