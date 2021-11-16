
import itertools
import random

from collections import deque
import time
import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")
import pickle
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, cpu_count

#import cvxpy as cp

from utils import redirect_log_file, Timer
#random.seed(1)


##################################################################

class Logic_Learning_Model():
    # Given one index of the head predicates
    # Return all the logic rules that will explain the occurrence rate of this head predicate
    # For example, the data set is generated from the following rules
    # A and B and Equal(A,B), and Before(A, D), then D;
    # C and Before(C, Not D), then  Not D
    # D Then  E, and Equal(D, E)
    # note that define the temporal predicates as compact as possible

    def __init__(self, head_predicate_idx = [4]):

        ### the following parameters are used to manually define the logic rules

        self.predicate_set= [0, 1, 2, 3] # the set of all meaningful predicates
        self.predicate_notation = ['A','B', 'C', 'D']
        self.static_pred_set = []
        self.instant_pred_set = []
        self.survival_pred_set = []
        self.body_pred_set = []
        self.body_pred_set_first_part = []
        self.body_pred_set_second_part = []
        self.prior_preds = []
        self.head_predicate_set = head_predicate_idx.copy()  # the index set of only one head predicates

        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.STATIC = "STATIC"
        self.num_formula = 0
        
        self.deleted_rules = set()
        self.low_grad_rules = dict()

        self.start_time = time.time()
        self.max_time = 2 * 3600 #2 hours

        # tunable params
        self.time_window = 10
        self.Time_tolerance = 0.1
        self.integral_resolution = 0.1
        self.decay_rate = 1
        self.batch_size = 64
        self.num_batch_check_for_feature = 1
        self.num_batch_check_for_gradient = 20
        self.num_batch_no_update_limit_opt = 300
        #self.num_batch_no_update_limit_ucb = 4
        self.num_iter  = 5
        self.num_iter_final = 20
        self.epsilon = 0.003
        self.gain_threshold = 0.02
        self.low_grad_threshold = 0.01
        self.low_grad_tolerance = 2
        self.weight_threshold = 0.01
        self.strict_weight_threshold = 0.1
        self.learning_rate = 0.001
        self.base_lr = 0.00005
        self.weight_lr = 0.005
        self.max_rule_body_length = 3 #
        self.max_num_rule = 20
        self.batch_size_cp = 500 # batch size used in cp. If too large, may out of memory.
        self.batch_size_grad = 500 #batch_size used in optimize_log_grad.
        #self.batch_size_init_ucb = 5
        #self.explore_rule_num_ucb = 8
        #self.explore_batch_size_ucb = 500
        self.use_cp = False
        self.worker_num = 8
        self.opt_worker_num = 12
        self.best_N = 1
        self.static_pred_coef = 1 #coef to balance static preds.
        self.debug_mode = False
        self.use_exp_kernel = False
        self.scale = 1
        self.use_decay = True
        self.use_2_bases = True
        self.init_base = 0.2
        self.init_weight = 0.1
        self.l1_coef = 0.1
        self.l2_coef = 0.1
        self.init_params()
        

    def init_params(self):
        #claim parameters and rule set
        self.model_parameter = {}
        self.logic_template = {}

        if self.use_exp_kernel:
            init_base = -abs(self.init_base)
        else:
            init_base = self.init_base

        for idx in self.head_predicate_set:
            self.model_parameter[idx] = {}
            if self.use_2_bases:
                self.model_parameter[idx]['base_0_1'] = torch.autograd.Variable((torch.ones(1) * init_base).double(), requires_grad=True)
                self.model_parameter[idx]['base_1_0'] = torch.autograd.Variable((torch.ones(1) * init_base).double(), requires_grad=True)
                if idx in self.survival_pred_set:
                    self.model_parameter[idx]['base_0_1'] = torch.autograd.Variable((torch.zeros(1)).double(), requires_grad=False)
            else:
                self.model_parameter[idx]['base'] = torch.autograd.Variable((torch.ones(1) * init_base).double(), requires_grad=True)
            #
            #self.model_parameter[idx]['base_cp'] = cp.Variable(1)
            self.logic_template[idx] = {}

    def print_info(self):
        print("-----key model information----")
        for valuename, value in vars(self).items():
            if isinstance(value, float) or isinstance(value, int) or isinstance(value, list):
                print("{}={}".format(valuename, value))
        print("----",flush=1)

    def get_model_parameters(self, head_predicate_idx):
        # collect all parameters in a list, used as input of Adam optimizer.
        parameters = list()
        if self.use_2_bases:
            parameters.append(self.model_parameter[head_predicate_idx]['base_0_1'])
            parameters.append(self.model_parameter[head_predicate_idx]['base_1_0'])
        else:
            parameters.append(self.model_parameter[head_predicate_idx]['base'])
        for formula_idx in range(self.num_formula): #TODO:potential bug
            parameters.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
        return parameters
    
    def share_memory(self, head_predicate_idx):
        params = self.get_model_parameters(head_predicate_idx)
        for p in params:
            p.share_memory_()


    
    def get_model_parameters_cp(self, head_predicate_idx):
        # collect all cp parameters in a list, used in l1 panelty
        parameters = list()
        parameters.append(self.model_parameter[head_predicate_idx]['base_cp'])
        for formula_idx in range(self.num_formula): #TODO:potential bug
            parameters.append(self.model_parameter[head_predicate_idx][formula_idx]['weight_cp'])
        return parameters
    
    def synchronize_cp_weight_with_torch(self,head_predicate_idx):
        params = self.get_model_parameters(head_predicate_idx)
        params_array = [p.item() for p in params]
        self.set_model_parameters_cp(head_predicate_idx, params_array)

    def synchronize_torch_weight_with_cp(self,head_predicate_idx):
        params = self.get_model_parameters_cp(head_predicate_idx)
        params_array = [p.value[0] for p in params]
        self.set_model_parameters(head_predicate_idx, params_array)

    def set_model_parameters(self, head_predicate_idx, param_array):
        # set model params
        parameters = list()
        if self.use_2_bases:
            base_0_1 = param_array[0]
            base_1_0 = param_array[1]
            self.model_parameter[head_predicate_idx]['base_0_1'] = torch.autograd.Variable((torch.ones(1) * base_0_1).double(), requires_grad=True)
            self.model_parameter[head_predicate_idx]['base_1_0'] = torch.autograd.Variable((torch.ones(1) * base_1_0).double(), requires_grad=True)
        else:
            base = param_array[0]
            self.model_parameter[head_predicate_idx]['base'] = torch.autograd.Variable((torch.ones(1) * base).double(), requires_grad=True)
        for formula_idx in range(self.num_formula):
            weight = param_array[formula_idx+1]
            self.model_parameter[head_predicate_idx][formula_idx]['weight'] = torch.autograd.Variable((torch.ones(1) * weight).double(), requires_grad=True)
    
    def set_model_parameters_cp(self, head_predicate_idx, param_array):
        # set model params
        parameters = list()
        base = cp.Variable(1)
        base.value = [param_array[0]]
        self.model_parameter[head_predicate_idx]['base_cp'] = base
        for formula_idx in range(self.num_formula):
            weight = cp.Variable(1)
            weight.value = [param_array[formula_idx+1]]
            self.model_parameter[head_predicate_idx][formula_idx]['weight_cp'] = weight
        
    def delete_rules(self, head_predicate_idx, formula_idx_list):
        # delete formulas listed in formula_idx_list
        #TODO: only consider single head

        # add rule_str to self.deleted_rules:
        for formula_idx in formula_idx_list:
            rule_str = self.get_rule_str(self.logic_template[head_predicate_idx][formula_idx], head_predicate_idx)
            self.deleted_rules.add(rule_str)

        # delete weight and logic-template
        tmp_logic_template = dict()
        tmp_model_parameter = dict()
        
        
        if self.use_2_bases:
            tmp_model_parameter['base_0_1'] = self.model_parameter[head_predicate_idx]['base_0_1']
            tmp_model_parameter['base_1_0'] = self.model_parameter[head_predicate_idx]['base_1_0']
        else:
            tmp_model_parameter['base'] = self.model_parameter[head_predicate_idx]['base']
        #tmp_model_parameter['base_cp'] = self.model_parameter[head_predicate_idx]['base_cp']
        
        new_formula_idx = 0
        for formula_idx in range(self.num_formula):
            if not formula_idx in formula_idx_list:
                tmp_logic_template[new_formula_idx] = self.logic_template[head_predicate_idx][formula_idx]
                tmp_model_parameter[new_formula_idx] = dict()
                tmp_model_parameter[new_formula_idx]["weight"] = self.model_parameter[head_predicate_idx][formula_idx]['weight']
                #tmp_model_parameter[new_formula_idx]["weight_cp"] = self.model_parameter[head_predicate_idx][formula_idx]['weight_cp']
                new_formula_idx += 1

        self.logic_template[head_predicate_idx] = tmp_logic_template
        self.model_parameter[head_predicate_idx] = tmp_model_parameter
        self.num_formula -= len(formula_idx_list)



    def intensity(self, cur_time, head_predicate_idx, dataset, sample_ID):
        feature_formula = []
        weight_formula = []
        effect_formula = []

        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            weight_formula.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
            
            f = self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx, history= dataset[sample_ID], template=self.logic_template[head_predicate_idx][formula_idx])
            feature_formula.append(f)
            effect_formula.append(self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                       history=dataset[sample_ID], template=self.logic_template[head_predicate_idx][formula_idx]))
        #print("feature_formula:", feature_formula)
        #print("effect_formula:", effect_formula)
        #print("weight_formula:", weight_formula)
        if len(weight_formula)>0:
            #intensity = torch.exp(torch.cat(weight_formula, dim=0))/torch.sum(torch.exp(torch.cat(weight_formula, dim=0)), dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)
            #NOTE: Softmax on weight leads to error when len(weight) = 1. Gradient on weight is very small.
            intensity =  torch.cat(weight_formula, dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)
            #print("intensity_raw(t={:.4f})={:.4f}".format(cur_time,intensity.data[0]))
        else:
            intensity = torch.zeros(1)

        if self.use_2_bases:
            state = self.get_state(cur_time, head_predicate_idx, dataset[sample_ID])
            if state == 1:
                base = self.model_parameter[head_predicate_idx]['base_1_0']
            elif state == 0:
                #print("state=0, use base_0_1")
                base = self.model_parameter[head_predicate_idx]['base_0_1']
            else:
                raise ValueError

        else:
            base = self.model_parameter[head_predicate_idx]['base']
        
        #print("inetensity before map:", intensity)
        if self.use_exp_kernel:
            # constraint base in [-1, 1], avoid overflow
            intensity = base + torch.sum(intensity)
            intensity = torch.exp(intensity) + 1e-3
        else:
            intensity = base + torch.sum(intensity)
            intensity = torch.nn.functional.relu(intensity) + 1e-3
        
        #print("inetensity after map:", intensity)
        

        return intensity
    
    def intensity_cp(self, cur_time, head_predicate_idx, history, mapping=True):
        intensity = np.zeros(1)

        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            w = self.model_parameter[head_predicate_idx][formula_idx]['weight_cp']
            f = self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,history=history, template=self.logic_template[head_predicate_idx][formula_idx])
            fe =self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,history=history, template=self.logic_template[head_predicate_idx][formula_idx])
            intensity += f.item() * fe.item() * w

        intensity += self.model_parameter[head_predicate_idx]['base_cp']
        
        if mapping:
            intensity = cp.exp(intensity)

        return intensity

    def get_feature(self, cur_time, head_predicate_idx, history, template):
        transition_time_dic = {}
        feature = torch.tensor([0], dtype=torch.float64)
        for idx, body_predicate_idx in enumerate(template['body_predicate_idx']):
            transition_time = np.array(history[body_predicate_idx]['time'])
            transition_state = np.array(history[body_predicate_idx]['state'])
            time_window_idx = (transition_time < cur_time) * (transition_time >= cur_time-self.time_window)
            transition_time = transition_time[time_window_idx]
            transition_state = transition_state[time_window_idx]
            #print("[body_predicate_idx, head_predicate_idx]=", [body_predicate_idx, head_predicate_idx])
            #print("template['temporal_relation_idx']=", template['temporal_relation_idx'])
            if (body_predicate_idx, head_predicate_idx) in template['temporal_relation_idx']:
                #print("early filter")
                #for time-relation with target, filter events by time-relation
                temporal_idx = template['temporal_relation_idx'].index((body_predicate_idx, head_predicate_idx))
                temporal_relation_type = template['temporal_relation_type'][temporal_idx]
                if  temporal_relation_type == self.BEFORE:
                    mask =  (transition_time < cur_time - self.Time_tolerance) * (transition_state == template['body_predicate_sign'][idx])
                elif temporal_relation_type == self.EQUAL:
                    mask = (transition_time >= cur_time - self.Time_tolerance) * (transition_state == template['body_predicate_sign'][idx])
                elif temporal_relation_type == self.STATIC:
                    mask = (transition_state == template['body_predicate_sign'][idx])
                else:
                    raise ValueError
            else:
                mask = transition_state == template['body_predicate_sign'][idx]

            transition_time_dic[body_predicate_idx] = transition_time[mask]
            if body_predicate_idx in self.static_pred_set and len(transition_time_dic[body_predicate_idx])>1:
                #for static variable, only keep the lastest matched result.
                transition_time_dic[body_predicate_idx] = np.array([transition_time_dic[body_predicate_idx][-1],]) 
            
        #transition_time_dic[head_predicate_idx] = [cur_time]
        ### get weights
        #print(transition_time_dic)
        # compute features whenever any item of the transition_item_dic is nonempty
        min_hist_len = min([len(i) for i in transition_time_dic.values()])
        if min_hist_len > 0:
            # need to compute feature using logic rules
            time_combination = np.array(list(itertools.product(*transition_time_dic.values())))
            time_combination_dic = {}
            for i, idx in enumerate(list(transition_time_dic.keys())):
                time_combination_dic[idx] = time_combination[:, i]
            temporal_kernel = np.ones(len(time_combination))
            for idx, temporal_relation_idx in enumerate(template['temporal_relation_idx']):
                time_0 = time_combination_dic[temporal_relation_idx[0]]
                if temporal_relation_idx[1] == head_predicate_idx:
                    time_1 = cur_time
                else:
                    time_1 = time_combination_dic[temporal_relation_idx[1]]
                time_difference = time_0 - time_1
                if template['temporal_relation_type'][idx] == self.BEFORE:
                    temporal_kernel *= (time_difference < - self.Time_tolerance) 
                    if self.use_decay:
                        temporal_kernel *= np.exp(-self.decay_rate *(cur_time - time_combination_dic[temporal_relation_idx[0]]))
                elif template['temporal_relation_type'][idx] == self.EQUAL:
                    temporal_kernel *= (- self.Time_tolerance <= time_difference) * (time_difference < 0)
                    if self.use_decay:
                        temporal_kernel *=  np.exp(-self.decay_rate*(cur_time - time_combination_dic[temporal_relation_idx[0]]))
                elif template['temporal_relation_type'][idx] == self.STATIC:
                    static_pred_idx = temporal_relation_idx[0]
                    if static_pred_idx in [0,1,2,3,4,5,]: #hard code for crime dataset.
                        # these are daily updating static variables.
                        static_time_tolerance = 24 * self.scale
                    else: # the others are hourly updating static variables.
                        static_time_tolerance = 1 * self.scale
                    temporal_kernel *= (- static_time_tolerance <= time_difference) * (time_difference < 0) 
                    #static relation are treated as 1, without decay. 
            feature = torch.tensor([np.sum(temporal_kernel)], dtype=torch.float64)
            if self.use_decay:
                feature *= self.decay_rate #this decay is important for convergence, see bug#113
        #if self.debug_mode:
        #    print("rule is : ", self.get_rule_str(rule=template, head_predicate_idx=head_predicate_idx) )
        #    print("feature at t={} is {}".format( cur_time,feature), flush=1)
        return feature

    def get_state(self, cur_time, head_predicate_idx, history):
        if head_predicate_idx in self.instant_pred_set:
            #instant pred state is always zero.
            #print("head_predicate_idx in self.instant_pred_set")
            cur_state = 0
            return cur_state

        head_transition_time = np.array(history[head_predicate_idx]['time'])
        head_transition_state = np.array(history[head_predicate_idx]['state'])
        
        if head_predicate_idx in self.survival_pred_set:
            default_state = 1
        else:
            default_state = 0
        
        if len(head_transition_time) == 0:
            cur_state = default_state
        else:
            idx = np.sum(cur_time > head_transition_time) - 1
            if idx < 0:
                cur_state = default_state
            else:
                cur_state = head_transition_state[idx]
        if head_predicate_idx in self.survival_pred_set and cur_state == 0:
            print(cur_time)
            print(head_transition_time)
            print(head_transition_state)

        return cur_state

    def get_formula_effect(self, cur_time, head_predicate_idx, history, template):
        ## Note this part is very important!! For generator, this should be np.sum(cur_time > head_transition_time) - 1
        ## Since at the transition times, choose the intensity function right before the transition time
        cur_state = self.get_state(cur_time, head_predicate_idx, history)

        counter_state = 1 - cur_state
        #print("counter_state=", counter_state)
        #print("template['head_predicate_sign']=", template['head_predicate_sign'])
        if counter_state - template['head_predicate_sign'][0] == 0: #fatal Error!!!
            formula_effect = torch.tensor([1], dtype=torch.float64)
        else:
            formula_effect = torch.tensor([-1], dtype=torch.float64)
        #print(formula_effect)
        #raise ValueError
        return formula_effect

    ### the following functions are for optimizing the logic weights
    def log_likelihood(self, head_predicate_idx, dataset, sample_ID_batch):
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        # iterate over samples
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            if head_predicate_idx in self.survival_pred_set:
                T_max = data_sample[head_predicate_idx]["time"][-1]
            else:
                T_max = max([data_sample[p]["time"][-1] if data_sample[p]["time"] else 0 for p in self.predicate_set])
            intensity_log_sum = self.intensity_log_sum(head_predicate_idx, dataset, sample_ID)
            intensity_integral = self.intensity_integral(head_predicate_idx, dataset, sample_ID)
            if self.debug_mode:
                print("intensity_log_sum=", intensity_log_sum)
                print("intensity_integral=", intensity_integral)
            log_likelihood += (intensity_log_sum - intensity_integral)
        if self.debug_mode:
            print("log_likelihood=",log_likelihood)
        return log_likelihood
    
    def log_likelihood_cp(self, head_predicate_idx, dataset, sample_ID_batch):
        log_likelihood = np.zeros(1)
        for sample_ID in sample_ID_batch:
            data_sample = dataset[sample_ID]
            intensity_log_sum = self.intensity_log_sum_cp(head_predicate_idx, data_sample)
            intensity_integral = self.intensity_integral_cp(head_predicate_idx, data_sample)
            log_likelihood += intensity_log_sum - intensity_integral
        return log_likelihood

    def intensity_log_sum(self, head_predicate_idx, dataset, sample_ID):
        intensity_transition = []
        if head_predicate_idx in self.instant_pred_set:
            trans_time = np.array(dataset[sample_ID][head_predicate_idx]['time'])
            state = np.array(dataset[sample_ID][head_predicate_idx]['state'])
            trans_time = trans_time[state==1]
        else:
            trans_time = dataset[sample_ID][head_predicate_idx]['time'][:]
        for t in trans_time:
            cur_intensity = self.intensity(t, head_predicate_idx, dataset, sample_ID)
            intensity_transition.append(cur_intensity)
        if len(intensity_transition) == 0: # only survival term, not event happens
            log_sum = torch.tensor([0], dtype=torch.float64)
        else:
            log_sum = torch.sum(torch.log(torch.cat(intensity_transition, dim=0)))
        #print(intensity_transition)
        return log_sum
    
    def intensity_log_sum_cp(self, head_predicate_idx, data_sample):
        log_sum = np.zeros(1)
        for t in data_sample[head_predicate_idx]['time'][:]:
            # directly calculate log-intensity, to avoid DCPError of cvxpy.
            log_intensity = self.intensity_cp(t, head_predicate_idx, data_sample, mapping=False)
            log_sum += log_intensity
        return log_sum

    def intensity_integral(self, head_predicate_idx, dataset, sample_ID):
        start_time = 0
        if head_predicate_idx in self.survival_pred_set:
            T_max = dataset[sample_ID][head_predicate_idx]["time"][-1]
        else:
            T_max = max([dataset[sample_ID][p]["time"][-1] if dataset[sample_ID][p]["time"] else 0 for p in self.predicate_set])
        end_time = T_max
        intensity_grid = []
        for t in np.arange(start_time, end_time, self.integral_resolution):
            cur_intensity = self.intensity(t, head_predicate_idx, dataset, sample_ID)
            intensity_grid.append(cur_intensity)
        if len(intensity_grid) == 0:
            return torch.tensor([0], dtype=torch.float64)
        integral = torch.sum(torch.cat(intensity_grid, dim=0)) * self.integral_resolution
        return integral


    def optimize_log_likelihood(self, head_predicate_idx, dataset, verbose=True):
        print("---- start optimize_log_likelihood ----", flush=1)
        print("Rule weights are:")
        self.print_rule()
        params = self.get_model_parameters(head_predicate_idx)
        optimizer = optim.Adam(params, lr=self.learning_rate)
        log_likelihood_batch = deque(list(), maxlen=self.num_batch_check_for_gradient)
        gradient_batch = deque(list(), maxlen=self.num_batch_check_for_gradient)
        params_batch = deque(list(), maxlen=self.num_batch_check_for_gradient)
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        best_log_likelihood = - 1e10
        num_batch_no_update = 0
        gradient_norm = 1e10
        epsilon = self.epsilon
        num_batch_run = 0

        for i in range(self.num_iter):
            if verbose:
                if i>0 and i%3==0:
                    print("{} th iter".format(i), flush=1)
                    print("grad norm={}. num_batch_no_update ={}".format(gradient_norm, num_batch_no_update))
                    self.print_rule()
            with Timer("{} th iter".format(i)) as t:
                sample_ID_list = list(dataset.keys())
                random.shuffle(sample_ID_list) #SGD
                #print("len(sample_ID_list)=",len(sample_ID_list))
                #print("num batches:", len(sample_ID_list)//self.batch_size)
                for batch_idx in range(len(sample_ID_list)//self.batch_size):
                    num_batch_run += 1
                    sample_ID_batch = sample_ID_list[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
                    optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                    log_likelihood = self.log_likelihood(head_predicate_idx, dataset, sample_ID_batch)
                    l_1 = torch.sum(torch.abs(torch.stack(params)))
                    loss = - log_likelihood + l_1
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    log_likelihood_batch.append(log_likelihood.data[0])
                    avg_log_likelihood = np.mean(log_likelihood_batch)
                    if avg_log_likelihood > best_log_likelihood:
                        best_log_likelihood = avg_log_likelihood
                        num_batch_no_update = 0
                    else:
                        num_batch_no_update +=1

                    batch_gradient = torch.autograd.grad(loss, params) # compute the batch gradient
                    batch_gradient = torch.stack(batch_gradient).detach().numpy()
                    #batch_gradient = np.min(np.abs(batch_gradient))
                    gradient_batch.append(batch_gradient)
                    gradient = np.mean(gradient_batch, axis=0)/self.batch_size # check the last N number of batch's gradient
                    gradient /= len(gradient) #bug#44, average gradient for multiple rules.
                    gradient_norm = np.linalg.norm(gradient)
                    #gradient_norm = gradient
                    params_detached = self.get_model_parameters(head_predicate_idx)
                    params_detached = torch.stack(params_detached).detach().numpy()
                    params_batch.append(params_detached)
                    #print('Screening now, the moving avg batch gradient norm is', gradient_norm, flush=True)
                    if len(gradient_batch) >= self.num_batch_check_for_gradient and (gradient_norm <= epsilon or num_batch_no_update >= self.num_batch_no_update_limit_opt):
                        break
            if len(gradient_batch) >= self.num_batch_check_for_gradient and (gradient_norm <= epsilon or num_batch_no_update >= self.num_batch_no_update_limit_opt):
                break
        print("Run {} batches".format(num_batch_run))
        if len(gradient_batch) >= self.num_batch_check_for_gradient and (gradient_norm <= epsilon or num_batch_no_update >= self.num_batch_no_update_limit_opt):
            print("grad norm {} <= epsilon {}. OR, num_batch_no_update {} >= num_batch_no_update_limit_opt {}".format(gradient_norm, epsilon, num_batch_no_update, self.num_batch_no_update_limit_opt))
        else:
            print("reach max iter num.")
            print("grad norm={}. num_batch_no_update ={}".format(gradient_norm, num_batch_no_update))
        #use the avg of last several batches log_likelihood
        log_likelihood = np.mean(log_likelihood_batch)/self.batch_size
        print('Finish optimize_log_likelihood, the log likelihood is', log_likelihood)
        #print("gradient_norm is ", gradient_norm)
        param_array = np.mean(params_batch, axis=0).reshape(-1)
        
        self.set_model_parameters(head_predicate_idx, param_array)
        
        #raise ValueError
        
        print("optimized rule weights are:")
        self.print_rule()
        print("---- exit optimize_log_likelihood ----", flush=1)
        print("--------",flush=1)
        
        return log_likelihood

    def _optimize_log_likelihood_mp_worker(self, head_predicate_idx, dataset, sample_ID_batch_list):
        params = self.get_model_parameters(head_predicate_idx)
        
        
        if self.use_2_bases:
            base_0_1_params = {"params": params[0], "lr":self.base_lr}
            base_1_0_params = {"params": params[1], "lr":self.base_lr}
            weight_params = {"params": params[2:], "lr":self.weight_lr}
            params_dicts = [base_0_1_params, base_1_0_params, weight_params] 
        else:
            base_params = {"params": params[:1], "lr":self.base_lr}
            weight_params = {"params": params[1:], "lr":self.weight_lr}
            params_dicts = [base_params, weight_params] 
        optimizer = optim.SGD(params_dicts, lr=self.learning_rate, weight_decay=self.l2_coef)

        for batch_idx, sample_ID_batch in enumerate(sample_ID_batch_list):
            # update weitghs
            optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
            
            log_likelihood = self.log_likelihood(head_predicate_idx, dataset, sample_ID_batch)
            l1 = torch.sum(torch.abs(torch.cat(params))) #l1 loss on weights and base.
            loss = - log_likelihood + self.l1_coef * l1

            
            if self.debug_mode:
                print("Before update batch_idx =", batch_idx)
                print("log_likelihood = ", log_likelihood.data[0])
                print("loss=", loss)
                for idx,p in enumerate(params):
                    print("param-{}={}".format(idx, p.data))
                    print("grad-{}=".format(idx), p.grad,flush=1)
            loss.backward()
            optimizer.step()

            if self.debug_mode:
                print("After update batch_idx=", batch_idx)
                for idx,p in enumerate(params):
                    print("param-{}={}".format(idx, p.data))
            time_cost = time.time() - self.start_time 
            print("time(s), log_likelihood = ", time_cost, log_likelihood.data.item(), flush=True)
            if time_cost > self.max_time:
                print("Exit due to exceeding maxinum time ", self.max_time)
                exit()
        if self.debug_mode:
            for idx,p in enumerate(params):
                print("param-{}={}".format(idx, p.data))
        
        
        return log_likelihood.detach().numpy()



    def optimize_log_likelihood_mp(self, head_predicate_idx, dataset, verbose=True):
        print("---- start optimize_log_likelihood multi-process----", flush=1)
        print("Rule set is:")
        self.print_rule()
        
        
        batch_num = len(dataset.keys())// self.batch_size
        batch_list= list()
        for i in range(self.num_iter):
            sample_ID_list = list(dataset.keys())
            random.shuffle(sample_ID_list) #random dataset order
            for batch_idx in range(batch_num):
                sample_ID_batch = sample_ID_list[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
                batch_list.append(sample_ID_batch)

        worker_num = min(self.opt_worker_num, cpu_count())
        worker_num = min(worker_num, len(batch_list))
        print("use {} workers for {} batches".format(worker_num, len(batch_list)))
        arg_list = list()
        batch_per_worker = len(batch_list)//worker_num
        for w in range(worker_num):
            sample_ID_batch_list = batch_list[w*batch_per_worker: (w+1)*batch_per_worker]
            arg_list.append((head_predicate_idx, dataset, sample_ID_batch_list))
        
        if worker_num > 1:
            self.share_memory(head_predicate_idx) #very important! share varibales across procs.
            with Pool(worker_num) as p:
                ret = p.starmap(self._optimize_log_likelihood_mp_worker, arg_list)
        else:
            ret = [self._optimize_log_likelihood_mp_worker(*arg) for arg in arg_list]
  
        log_likelihood_list = ret
        log_likelihood = np.mean(log_likelihood_list)/self.batch_size
        
        print("optimized rule weights are:")
        self.print_rule()
        print("---- exit optimize_log_likelihood multi-process----", flush=1)
        
        return log_likelihood


    ### the following 2 functions are to compute sub-problem objective function
    def intensity_log_gradient(self, head_predicate_idx, dataset, sample_ID):
        # 
        if self.use_exp_kernel:
            return torch.tensor([1.0],dtype=torch.float64) #intensity_log_gradient of exp kernel is always 1.
        else:
            intensity_transition = []
            for t in dataset[sample_ID][head_predicate_idx]['time'][:]:
                cur_intensity = self.intensity(t, head_predicate_idx, dataset, sample_ID)
                cur_intensity = cur_intensity.detach() #detach, since multiprocessing needs requires_grad=False
                intensity_transition.append(cur_intensity)
            if len(intensity_transition) == 0:  # only survival term, not event happens
                log_gradient = torch.tensor([0], dtype=torch.float64)
            else:
                log_gradient = torch.cat(intensity_transition, dim=0).pow(-1)
            return log_gradient


    
    def intensity_integral_gradient(self, head_predicate_idx, dataset, sample_ID):

        if self.use_exp_kernel:
            start_time = 0
            if head_predicate_idx in self.survival_pred_set:
                T_max = dataset[sample_ID][head_predicate_idx]["time"][-1]
            else:
                T_max = max([dataset[sample_ID][p]["time"][-1] if dataset[sample_ID][p]["time"] else 0 for p in self.predicate_set])
            end_time = T_max
            intensity_gradient_grid = []
            for t in np.arange(start_time, end_time, self.integral_resolution):
                cur_intensity = self.intensity(t, head_predicate_idx, dataset, sample_ID)
                cur_intensity = cur_intensity.detach() #detach, since multiprocessing needs requires_grad=False
                intensity_gradient_grid.append(cur_intensity)   # due to that the derivative of exp(x) is still exp(x)
            if len(intensity_gradient_grid)==0:
                return torch.tensor([0],dtype=torch.float64)
            integral_gradient_grid = torch.cat(intensity_gradient_grid, dim=0) * self.integral_resolution
            return integral_gradient_grid
        else:
            return torch.tensor([1],dtype=torch.float64) * self.integral_resolution


    def log_likelihood_gradient(self, head_predicate_idx, dataset,  intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template):
        #calculate gradient of likelihood at w_new = 0.

        log_likelihood_grad_list = list()
        # iterate over samples
        sample_ID_batch = list(dataset.keys())
        
        if len(sample_ID_batch) > self.batch_size_grad and self.batch_size_grad>0:
            #print("Random select {} samples from {} samples".format(self.batch_size_grad, len(sample_ID_batch)), flush=1)
            sample_ID_batch = random.sample(sample_ID_batch, self.batch_size_grad)
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            # compute the log_intensity_gradient, integral_gradient_grid using existing rules
            if head_predicate_idx in self.survival_pred_set:
                T_max = dataset[sample_ID][head_predicate_idx]["time"][-1]
            else:
                T_max = max([data_sample[p]["time"][-1] if data_sample[p]["time"] else 0  for p in self.predicate_set])
            start_time = 0
            end_time = T_max # Note for different sample_ID, the T_max can be different
            # compute new feature at the transition times
            new_feature_transition_times = []
            for t in data_sample[head_predicate_idx]['time'][:]:
                f = self.get_feature(cur_time=t, head_predicate_idx=head_predicate_idx, history=data_sample, template =new_rule_template)
                fe = self.get_formula_effect(cur_time=t, head_predicate_idx=head_predicate_idx, history=data_sample, template =new_rule_template)
                new_feature_transition_times.append(f * fe)
            new_feature_grid_times = []
            for t in np.arange(start_time, end_time, self.integral_resolution):
                 
                f = self.get_feature(cur_time=t, head_predicate_idx=head_predicate_idx, history=data_sample, template =new_rule_template)
                fe = self.get_formula_effect(cur_time=t, head_predicate_idx=head_predicate_idx, history=data_sample, template =new_rule_template)
                new_feature_grid_times.append(f * fe)

            if len(new_feature_transition_times)>0:
                new_feature_transition_times = torch.cat(new_feature_transition_times, dim=0)
            else:
                new_feature_transition_times = torch.zeros(1, dtype=torch.float64)

            if len(new_feature_grid_times) >0:
                new_feature_grid_times = torch.cat(new_feature_grid_times, dim=0)
            else:
                new_feature_grid_times = torch.zeros(1, dtype=torch.float64)

            sum_intensity_log_grad = torch.sum(intensity_log_gradient[sample_ID] * new_feature_transition_times)
            intensity_integral_grad = torch.sum(intensity_integral_gradient_grid[sample_ID] * new_feature_grid_times, dim=0)
            log_likelihood_grad =  sum_intensity_log_grad - intensity_integral_grad
            #print("sample-{}: term1={}, term2={}, log-grad={}".format(sample_ID, sum_intensity_log_grad.item(), intensity_integral_grad.item(), log_likelihood_grad.item()))
                                       
            log_likelihood_grad_list.append(log_likelihood_grad)
        mean_grad = np.mean(log_likelihood_grad_list) + self.l1_coef 
        std_grad = np.std(log_likelihood_grad_list,ddof=1) #ddof=1 for unbiased estimation
        #note: we assume l1 penalty gradient is zero, at w=0
        return mean_grad, std_grad


    def optimize_log_likelihood_gradient(self, head_predicate_idx, dataset,  intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template):
        # in old codes, this function optimizes time relation params,
        # now there is no time relation params, so no optimization.
        gain = self.log_likelihood_gradient(head_predicate_idx, dataset,  intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template)
        return gain



    # here we use the  width-first search to add body predicates
    # we assume that the number of important rules should be smaller thant K
    # we assume that the length of the body predicates should be smaller thant L          
            

    def check_repeat(self, new_rule, head_predicate_idx):
        new_rule_body_predicate_set = set(new_rule['body_predicate_idx'])
        new_rule_temporal_relation_set = set(zip(new_rule['temporal_relation_idx'], new_rule['temporal_relation_type']))
        for rule in self.logic_template[head_predicate_idx].values():
            # strict check repeat, ignoring sign, due to bug#77
            body_predicate_set = set(rule['body_predicate_idx'])
            if body_predicate_set == new_rule_body_predicate_set:
                temporal_relation_set = set(zip(rule['temporal_relation_idx'], rule['temporal_relation_type']))
                if temporal_relation_set == new_rule_temporal_relation_set:
                    return True #repeat with existing rules
        rule_str = self.get_rule_str(new_rule, head_predicate_idx)
        if rule_str in self.deleted_rules:
            return True #repeat with deleted rules
        return False #not repeat


    def generate_rule_via_column_generation(self, head_predicate_idx, dataset):
        print("----- start generate_rule_via_column_generation -----")
        ## generate one new rule to the rule set by columns generation
        new_rule_table = {}
        new_rule_table[head_predicate_idx] = {}
        new_rule_table[head_predicate_idx]['body_predicate_idx'] = []
        new_rule_table[head_predicate_idx]['body_predicate_sign'] = []  # use 1 to indicate True; use 0 to indicate False
        new_rule_table[head_predicate_idx]['head_predicate_sign'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_idx'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_type'] = []
        new_rule_table[head_predicate_idx]['performance_gain'] = []


        #calculate intensity for sub-problem
        intensity_log_gradient, intensity_integral_gradient_grid = self.get_intensity_and_integral_grad( head_predicate_idx, dataset)
        ## search for the new rule from by minimizing the gradient of the log-likelihood
        arg_list = list()
        print("start enumerating candidate rules.", flush=1)
        for head_predicate_sign in [1,]:
            if head_predicate_idx in self.instant_pred_set and head_predicate_sign == 0:
                # instant pred should not be negative
                continue
            for body_predicate_sign in [1, ]: # consider head_predicate_sign = 1/0
                if self.body_pred_set_first_part:
                    candidate_body_pred_set = self.body_pred_set_first_part
                else:
                    candidate_body_pred_set = self.body_pred_set
                for body_predicate_idx in candidate_body_pred_set:  
                    if body_predicate_idx in self.instant_pred_set and body_predicate_sign == 0:
                        # instant pred should not be negative
                        continue

                    if body_predicate_idx in self.static_pred_set: #do not allow static pred form rule alone.
                        continue
                    for temporal_relation_type in [self.BEFORE, self.EQUAL]:
                        # create new rule
                        new_rule_template = {}
                        new_rule_template[head_predicate_idx]= {}
                        new_rule_template[head_predicate_idx]['body_predicate_idx'] = [body_predicate_idx]
                        new_rule_template[head_predicate_idx]['body_predicate_sign'] = [body_predicate_sign]  # use 1 to indicate True; use 0 to indicate False
                        new_rule_template[head_predicate_idx]['head_predicate_sign'] = [head_predicate_sign]
                        new_rule_template[head_predicate_idx]['temporal_relation_idx'] = [(body_predicate_idx, head_predicate_idx)]
                        new_rule_template[head_predicate_idx]['temporal_relation_type'] = [temporal_relation_type]

                        if self.check_repeat(new_rule_template[head_predicate_idx], head_predicate_idx): # Repeated rule is not allowed.
                            continue
                
                        arg_list.append((head_predicate_idx, dataset, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template[head_predicate_idx]))

                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append(new_rule_template[head_predicate_idx]['body_predicate_idx'] )
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append(new_rule_template[head_predicate_idx]['body_predicate_sign'])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append(new_rule_template[head_predicate_idx]['head_predicate_sign'])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(new_rule_template[head_predicate_idx]['temporal_relation_idx'])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(new_rule_template[head_predicate_idx]['temporal_relation_type'])


        is_update_weight, is_continue, added_rule_str_list = self.select_and_add_new_rule(head_predicate_idx, arg_list, new_rule_table, dataset)
        
        print("----- exit generate_rule_via_column_generation -----",flush=1)
        return is_update_weight, is_continue, added_rule_str_list
            
    def get_intensity_and_integral_grad_worker(self, head_predicate_idx, dataset, sample_ID):
        intensity_log_gradient = self.intensity_log_gradient(head_predicate_idx, dataset, sample_ID)
        intensity_integral_gradient = self.intensity_integral_gradient(head_predicate_idx, dataset, sample_ID)
        return intensity_log_gradient, intensity_integral_gradient

    def get_intensity_and_integral_grad(self, head_predicate_idx, dataset):
        print("---start calculate intensity grad and integral grad.---", flush=1)
        with Timer("calculate intensity grad and integral grad") as t:
            sample_ID_batch = list(dataset.keys())
            intensity_log_gradient = dict()
            intensity_integral_gradient_grid = dict()
            arg_list = list()  #args for parallel intensity_integral_gradient_grid()
            for sample_ID in sample_ID_batch:
                arg_list.append((head_predicate_idx, dataset, sample_ID)) #args for parallel intensity_integral_gradient_grid()
            
            worker_num = min(self.worker_num, cpu_count())
            worker_num = min(worker_num, len(arg_list))
            print("use {} workers, run {} data".format(worker_num,len(arg_list)), flush=1 )
            if worker_num > 1:
                with Pool(worker_num) as p:
                    ret = p.starmap(self.get_intensity_and_integral_grad_worker, arg_list) 
            else:
                ret = [self.get_intensity_and_integral_grad_worker(*arg) for arg in arg_list]

            for idx, sample_ID in enumerate(sample_ID_batch):
                intensity_log_gradient[sample_ID] = ret[idx][0]
                intensity_integral_gradient_grid[sample_ID] = ret[idx][1]
        print("---exit calculate intensity grad and integral grad.---", flush=1)
        return intensity_log_gradient, intensity_integral_gradient_grid


    def add_one_predicate_to_existing_rule(self, head_predicate_idx, dataset, existing_rule_template):
        print("----- start add_one_predicate_to_existing_rule -----",flush=1)
        ## increase the length of the existing rules
        ## generate one new rule to the rule set by columns generation
        ## (most of codes are same with generate_rule_via_column_generation())
        new_rule_table = {}
        new_rule_table[head_predicate_idx] = {}
        new_rule_table[head_predicate_idx]['body_predicate_idx'] = []
        new_rule_table[head_predicate_idx]['body_predicate_sign'] = []
        new_rule_table[head_predicate_idx]['head_predicate_sign'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_idx'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_type'] = []
        new_rule_table[head_predicate_idx]['performance_gain'] = []

        #calculate intensity for sub-problem
        intensity_log_gradient, intensity_integral_gradient_grid = self.get_intensity_and_integral_grad( head_predicate_idx, dataset )



        ## search for the new rule from by minimizing the gradient of the log-likelihood
        #be careful, do NOT modify existing rule.
        arg_list = list()
        print("start enumerating candidate rules.", flush=1)
        
        for body_predicate_sign in [1, ]:
            if self.body_pred_set_second_part:
                candidate_body_pred_set = self.body_pred_set_second_part
            else:
                candidate_body_pred_set = self.body_pred_set
            for body_predicate_idx in candidate_body_pred_set:
                if body_predicate_idx in existing_rule_template['body_predicate_idx']: 
                    # repeated predicates are not allowed.
                    continue 
                if body_predicate_idx in self.instant_pred_set and body_predicate_sign == 0:
                    # instant pred should not be negative
                    continue
                
                
                
                if body_predicate_idx in self.static_pred_set:
                    time_relation_list = [self.STATIC]
                     # if pred_0 is static, then pred_1 should be non-static. Also, pred_1 should not be head.
                    existing_predicate_idx_list = existing_rule_template['body_predicate_idx']
                    candidate_predicate_list = [pred for pred in existing_predicate_idx_list if pred not in self.static_pred_set] 
                else:
                    # if pred_0 is non-static, then pred_1 should also be non-static
                    time_relation_list = [self.BEFORE, self.EQUAL]
                    existing_predicate_idx_list = [head_predicate_idx] + existing_rule_template['body_predicate_idx']
                    candidate_predicate_list = [pred for pred in existing_predicate_idx_list if pred not in self.static_pred_set] 
                
                candidate_predicate_list = list(set(candidate_predicate_list))
                
                
                for temporal_relation_type in time_relation_list:
                    for candidate_predicate_idx in candidate_predicate_list:
                        
                        # create new rule
                        new_rule_template = {}
                        new_rule_template[head_predicate_idx]= {}
                        new_rule_template[head_predicate_idx]['body_predicate_idx'] =  existing_rule_template['body_predicate_idx'] + [body_predicate_idx] 
                        new_rule_template[head_predicate_idx]['body_predicate_sign'] =  existing_rule_template['body_predicate_sign'] + [body_predicate_sign] 
                        new_rule_template[head_predicate_idx]['head_predicate_sign'] = existing_rule_template['head_predicate_sign'] + []
                        new_rule_template[head_predicate_idx]['temporal_relation_idx'] =  existing_rule_template['temporal_relation_idx'] + [(body_predicate_idx, candidate_predicate_idx)] 
                        new_rule_template[head_predicate_idx]['temporal_relation_type'] =  existing_rule_template['temporal_relation_type'] + [temporal_relation_type] 

                        if self.check_repeat(new_rule_template[head_predicate_idx], head_predicate_idx): # Repeated rule is not allowed.
                            continue
                        

                        arg_list.append((head_predicate_idx, dataset, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template[head_predicate_idx]))

                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append(new_rule_template[head_predicate_idx]['body_predicate_idx'] )
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append(new_rule_template[head_predicate_idx]['body_predicate_sign'])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append(new_rule_template[head_predicate_idx]['head_predicate_sign'])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(new_rule_template[head_predicate_idx]['temporal_relation_idx'])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(new_rule_template[head_predicate_idx]['temporal_relation_type'])
   

        is_update_weight, is_continue, added_rule_str_list = self.select_and_add_new_rule(head_predicate_idx, arg_list, new_rule_table, dataset)

        print("----- exit add_one_predicate_to_existing_rule -----",flush=1)
        return is_update_weight, is_continue, added_rule_str_list

    def select_and_add_new_rule(self, head_predicate_idx, arg_list, new_rule_table, dataset):
        
        print("----- start select_and_add_new_rule -----",flush=1)
        if len(arg_list) == 0: # No candidate rule generated.
            print("No candidate rule generated.")
            is_update_weight = False
            is_continue = False
            print("----- exit select_and_add_new_rule -----",flush=1)
            return is_update_weight, is_continue, []

        # all-data gradient
        print("-------start multiprocess------",flush=1)
        if self.batch_size_grad <=0:
            self.batch_size_grad = len(dataset.keys()) # run all samples for grad.
        cpu = cpu_count()
        worker_num = min(self.worker_num, cpu)
        worker_num = min(worker_num, len(arg_list))
        print("cpu num = {}, use {} workers, process {} candidate rules.".format(cpu, worker_num, len(arg_list)))
        with Timer("multiprocess log-grad") as t:
            
            if worker_num > 1: #multiprocessing
                
                with Pool(worker_num) as pool:
                    gain_all_data = pool.starmap(self.optimize_log_likelihood_gradient, arg_list)
            else: #single process, not use pool.
                gain_all_data = [self.optimize_log_likelihood_gradient(*arg) for arg in arg_list]
        mean_gain_all_data, std_gain_all_data  = list(zip(*gain_all_data))
        print("-------end multiprocess------",flush=1)
        
                
        #delete low gain candidate rules
        for idx, gain_ in enumerate(mean_gain_all_data):
            if abs(gain_) < self.low_grad_threshold:
                rule_str = self.get_rule_str(arg_list[idx][-1], head_predicate_idx)
                if rule_str in self.low_grad_rules:
                    self.low_grad_rules[rule_str] += 1
                else:
                    self.low_grad_rules[rule_str] = 1
                if self.low_grad_rules[rule_str] >= self.low_grad_tolerance:
                    # this low-grad rule repeat for too many times, delete it, and never re-visit it.
                    self.deleted_rules.add(rule_str)

        print("------Select N best rule-------")
        # choose the N-best rules that lead to the optimal log-likelihood
        #sorted_idx_gain_all_data = sorted(list(enumerate(mean_gain_all_data)), key=lambda x:x[1], reverse=True) # sort by mean gain, descending
        sorted_idx_gain_all_data = sorted(list(enumerate(mean_gain_all_data)), key=lambda x:abs(x[1]), reverse=True) # sort by mean absolute gain, descending
        if self.prior_preds:
            sorted_idx_gain_all_data = sorted(sorted_idx_gain_all_data, key=lambda x: sum([p in self.prior_preds for p in arg_list[x[0]][-1]['body_predicate_idx']]), reverse=True) # sort by prior preds, descending
            #print(sorted_idx_gain_all_data)
            #raise ValueError
        for idx, gain in sorted_idx_gain_all_data:
            rule_str = self.get_rule_str(arg_list[idx][-1], head_predicate_idx)
            std_gain = std_gain_all_data[idx]
            print("log-likelihood-grad(all-data) mean= {:.5f}, std={:.5f}, Rule = {}".format(gain, std_gain, rule_str))
            print("-------------", flush=True)
        is_update_weight = False
        is_continue = False
        added_rule_str_list = list()
        for i in range(self.best_N):
            if i >= len(sorted_idx_gain_all_data):
                break
            idx, best_gain = sorted_idx_gain_all_data[i]

            if  abs(best_gain) > self.gain_threshold:
            #if  best_gain > self.gain_threshold:
                # add new rule
                self.logic_template[head_predicate_idx][self.num_formula] = {}
                self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

                best_rule_str = self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula], head_predicate_idx)
                added_rule_str_list.append(best_rule_str)
                print("Best rule is:", best_rule_str)
                print("Best log-likelihood-grad(all-data) =", best_gain)
                # add model parameter
                self.model_parameter[head_predicate_idx][self.num_formula] = {}
                self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable((torch.ones(1) * self.init_weight).double(), requires_grad=True)
                #self.model_parameter[head_predicate_idx][self.num_formula]['weight_cp'] = cp.Variable(1)
                self.num_formula += 1
                is_update_weight = True
                is_continue = True
                print("new rule added.")
            else:
                is_continue = False
                print("best gain {} does not meet thershold {}.".format(best_gain, self.gain_threshold))
                break

        if is_update_weight:
            # update model parameter
            if not self.use_cp:
                with Timer("optimize log-likelihood (torch)") as t:
                    l = self.optimize_log_likelihood_mp(head_predicate_idx, dataset )
                print("Update Log-likelihood (torch)= ", l, flush=1)

            else:
                with Timer("optimize log-likelihood (cp)") as t:
                    l_cp = self.optimize_log_likelihood_cp(head_predicate_idx, dataset)
                print("Update Log-likelihood (cvxpy)= ", l_cp)

            print("Added rule and re-fitted weights. Current rule set is:")
            self.print_rule_cp()
            
            # prune after added.
            is_strict = self.num_formula >= self.max_num_rule #if too many rules, use strict threshold, otherwise, use normal threshold.
            is_pruned = self.prune_rules_with_small_weights(head_predicate_idx, dataset, is_strict)
            if is_pruned: #after prunning, maybe add more rules.
                is_continue = True
                #update added_rule_str_list, remove the deleted rule_str.
                added_rule_str_list = [rule_str for rule_str in added_rule_str_list if self.get_rule_idx(head_predicate_idx, rule_str) > -1]

        print("----- exit select_and_add_new_rule -----",flush=1)
        return is_update_weight, is_continue, added_rule_str_list


    def prune_rules_with_small_weights(self, head_predicate_idx, dataset, is_strict=False):
        formula_idx_list = list()
        if is_strict:
            thershold = self.strict_weight_threshold
        else:
            thershold = self.weight_threshold

        for formula_idx in range(self.num_formula):
            w = self.model_parameter[head_predicate_idx][formula_idx]['weight'].detach()
            #if w < thershold:
            if  abs(w) < thershold:
                formula_idx_list.append(formula_idx)
        if len(formula_idx_list) > 0:
            print("delete these rules:",formula_idx_list)
            self.delete_rules(head_predicate_idx, formula_idx_list)
            #refit weights
            print("start re-fit weights", flush=1)
            if not self.use_cp:
                with Timer("optimize log-likelihood (torch)") as t:
                    l = self.optimize_log_likelihood_mp(head_predicate_idx, dataset)
                print("Update Log-likelihood (torch)= ", l, flush=1)

            else:
                with Timer("optimize log-likelihood (cp)") as t:
                    l_cp = self.optimize_log_likelihood_cp(head_predicate_idx, dataset)
                print("Update Log-likelihood (cvxpy)= ", l_cp)
            
            print("Deleted some rules and refited weights, Current rule set is:")
            self.print_rule_cp()
            return True
        return False
            

    def BFS(self, head_predicate_idx, training_dataset, testing_dataset, tag, init_params=True):
        
        self.print_info()
        if init_params:
            self.init_params()
        with Timer("initial optimize") as t:
            l = self.optimize_log_likelihood_mp(head_predicate_idx, training_dataset)
            print("log-likelihood=",l)
        #self.generate_target( head_predicate_idx, testing_dataset, num_repeat=1)
        print("----- start BFS -----", flush=1)
        self.start_time = time.time()
        #Begin Breadth(width) First Search
        #generate new rule from scratch

        is_continue = True
        while self.num_formula < self.max_num_rule and is_continue:
            is_update_weight, is_continue, _ = self.generate_rule_via_column_generation(head_predicate_idx, training_dataset)
            #if is_update_weight:
            #    self.generate_target( head_predicate_idx, testing_dataset, num_repeat=1)
         
        #generate new rule by extending existing rules
        extended_rules = set()
        for cur_body_length in range(1, self.max_rule_body_length):
            flag = True
            while(self.num_formula < self.max_num_rule and flag):
                #select all existing rules whose length are cur_body_length
                idx_template_list = [(idx, template) for idx, template in self.logic_template[head_predicate_idx].items() if len(template['body_predicate_idx']) == cur_body_length]
                # sort by weights, descending
                idx_template_list  = sorted(idx_template_list, key=lambda x:self.model_parameter[head_predicate_idx][x[0]]['weight'].data[0], reverse=True) 
                
                # select best unextended rule to extend.
                template_to_extend = None
                for idx, template in idx_template_list:
                    rule_str = self.get_rule_str(template, head_predicate_idx)
                    if not rule_str in extended_rules:
                        template_to_extend = template
                        break

                if template_to_extend is None: #no template can be extended, break, go to search next length rule.
                    flag = False
                    break
                
                #extend the selected rule.
                rule_str = self.get_rule_str(template_to_extend, head_predicate_idx)
                print("start to extend this rule:", rule_str)
                is_update_weight , is_continue, _ = self.add_one_predicate_to_existing_rule(head_predicate_idx, training_dataset, template_to_extend)
                if not is_continue: # this rule is fully explored, don't re-visit it.
                    extended_rules.add(rule_str) 
                #if is_update_weight:
                #    self.generate_target( head_predicate_idx, testing_dataset, num_repeat=1)


        print("BFS finished, rule set is:")
        self.print_rule_cp()

        self.final_tune(head_predicate_idx, training_dataset)
        #self.generate_target( head_predicate_idx, testing_dataset, num_repeat=100)
        with open("./model/model-{}.pkl".format(tag),'wb') as f:
            pickle.dump(self, f)
        print("----- exit BFS -----", flush=1)

    def Hawkes(self, head_predicate_idx, training_dataset, testing_dataset, tag):
        print("----- start Hawkes -----", flush=1)
        INF = 10000000
        self.best_N = INF
        self.max_num_rule = INF
        self.max_rule_body_length = 1
        self.weight_threshold = -1
        self.strict_weight_threshold = -1
        self.gain_threshold = -1
        self.num_iter_final = 1
        self.batch_size_grad = 1
        self.body_pred_set_first_part = []
        self.body_pred_set_second_part = []
        self.static_pred_set = []
        self.instant_pred_set = self.body_pred_set
        self.BFS(head_predicate_idx, training_dataset, testing_dataset, tag)
        print("----- end Hawkes ----", flush = 1)
    def final_tune(self, head_predicate_idx, dataset, ):
        print("----- start final_tune -----", flush=1)
        # final prune, with strict threshold of weight.
        pruned = True
        while pruned:
            pruned = self.prune_rules_with_small_weights(head_predicate_idx, dataset, is_strict=True )
        
        # final optimize, with large num_iter (only for torch)
        if not self.use_cp:
            self.num_iter = self.num_iter_final
            self.optimize_log_likelihood_mp(head_predicate_idx, dataset)
        print("final_tune finished, rule set is:")
        self.print_rule_cp()
        print("----- exit final_tune -----", flush=1)
        

    def print_rule(self):
        for head_predicate_idx, rules in self.logic_template.items():
            
            if self.use_2_bases:
                print("Head:{}, base_0_1={:.4f}, base_1_0={:.4f},".format(self.predicate_notation[head_predicate_idx], self.model_parameter[head_predicate_idx]['base_0_1'].data[0], self.model_parameter[head_predicate_idx]['base_1_0'].data[0]))
            else:
                print("Head:{}, base={:.4f},".format(self.predicate_notation[head_predicate_idx], self.model_parameter[head_predicate_idx]['base'].data[0]))
            for rule_id, rule in rules.items():
                rule_str = "Rule{}: ".format(rule_id)
                rule_str += self.get_rule_str(rule, head_predicate_idx)
                weight = self.model_parameter[head_predicate_idx][rule_id]['weight'].data[0]
                rule_str += ", weight={:.4f},".format(weight)
                print(rule_str)
    
    def print_rule_cp(self):
        for head_predicate_idx, rules in self.logic_template.items():
            
            if 'base_cp' in self.model_parameter[head_predicate_idx] and self.model_parameter[head_predicate_idx]['base_cp'].value:
                base_cp = self.model_parameter[head_predicate_idx]['base_cp'].value
                print("Head:{}, base(cp)={:.4f},".format(self.predicate_notation[head_predicate_idx], base_cp[0]))
            else:
                if self.use_2_bases:
                    base_0_1 = self.model_parameter[head_predicate_idx]['base_0_1'].item()
                    base_1_0 = self.model_parameter[head_predicate_idx]['base_1_0'].item()
                    print("Head:{}, base_0_1(torch) = {:.4f}, base_1_0(torch) = {:.4f},".format(self.predicate_notation[head_predicate_idx], base_0_1, base_1_0))
                else:
                    base = self.model_parameter[head_predicate_idx]['base'].item()
                    print("Head:{}, base(torch) = {:.4f},".format(self.predicate_notation[head_predicate_idx], base))
            for rule_id, rule in rules.items():
                rule_str = "Rule{}: ".format(rule_id)
                rule_str += self.get_rule_str(rule, head_predicate_idx)
                weight = self.model_parameter[head_predicate_idx][rule_id]['weight'].item()
                if "weight_cp" in self.model_parameter[head_predicate_idx][rule_id] and self.model_parameter[head_predicate_idx][rule_id]['weight_cp'].value:
                    weight_cp = self.model_parameter[head_predicate_idx][rule_id]['weight_cp'].value
                    rule_str += ", weight(torch)={:.4f}, weight(cp)={:.4f},".format(weight, weight_cp[0])
                else:
                    rule_str += ", weight(torch)={:.4f},".format(weight)
                print(rule_str)
    
    def get_rule_str(self, rule, head_predicate_idx):
        body_predicate_idx = rule['body_predicate_idx']
        body_predicate_sign = rule['body_predicate_sign']
        head_predicate_sign = rule['head_predicate_sign'][0]
        temporal_relation_idx = rule['temporal_relation_idx']
        temporal_relation_type = rule['temporal_relation_type']
        rule_str = ""
        negative_predicate_list = list()
        for i in range(len(body_predicate_idx)):
            if body_predicate_sign[i] == 0:
                rule_str += "Not "
                negative_predicate_list.append(body_predicate_idx[i])
            rule_str += self.predicate_notation[body_predicate_idx[i]]
            if i <= len(body_predicate_idx) - 2:
                rule_str += " ^ "
        rule_str += " --> "
        if head_predicate_sign == 0:
            rule_str += "Not "
            negative_predicate_list.append(head_predicate_idx)
        rule_str += self.predicate_notation[head_predicate_idx]
        rule_str += " , "

        for i in range(len(temporal_relation_idx)):
            if temporal_relation_idx[i][0] in negative_predicate_list:
                rule_str += "Not "
            rule_str += self.predicate_notation[temporal_relation_idx[i][0]]
            rule_str += " {} ".format(temporal_relation_type[i])
            if temporal_relation_idx[i][1] in negative_predicate_list:
                rule_str += "Not "
            rule_str += self.predicate_notation[temporal_relation_idx[i][1]]
            if i <= len(temporal_relation_idx) - 2:
                rule_str += " ^ "   
        return rule_str     

    def get_rule_idx(self, head_predicate_idx, rule_str):
        for i in range(self.num_formula):
            rule_i = self.logic_template[head_predicate_idx][i]
            rule_str_i = self.get_rule_str(rule_i, head_predicate_idx)
            if rule_str_i == rule_str:
                return i #return matched formula_idx
        return -1 #if no such rule, return -1.

    def DFS(self,head_predicate_idx, training_dataset, testing_dataset, tag, init_params=True):
        self.print_info()
        if init_params:
            self.init_params()
        with Timer("initial optimize") as t:
            l = self.optimize_log_likelihood_mp(head_predicate_idx, training_dataset)
            print("log-likelihood=",l)
        #self.generate_target( head_predicate_idx, testing_dataset, num_repeat=1)
        print("----- start DFS -----", flush=1)
        self.start_time = time.time()
        rule_to_extend_str_stack = list() #use stack to implement DFS
        while self.num_formula < self.max_num_rule:
            if len(rule_to_extend_str_stack) == 0: #if stack empty,  generate len-1 rule
                print("--- DFS stack is empty, add len-1 rules ---")
                is_update_weight, is_continue, added_rule_str_list = self.generate_rule_via_column_generation(head_predicate_idx, training_dataset)
                for added_rule_str in added_rule_str_list:
                    rule_to_extend_str_stack.append(added_rule_str)
                if not is_continue: #no new rule added, DFS terminates.
                    break

            else: #extend existing rule
                print("--- try to extend existing rule.")
                print("--- DFS stack now contains:\n", ",\n".join(rule_to_extend_str_stack))
                rule_to_extend_str = rule_to_extend_str_stack[-1]
                rule_to_extend_idx = self.get_rule_idx(head_predicate_idx, rule_to_extend_str)
                if rule_to_extend_idx <= -1 or len(self.logic_template[head_predicate_idx][rule_to_extend_idx]['body_predicate_idx']) >= self.max_rule_body_length:
                    # if rule_ro_extend is deleted, or too long, then skip it and conitnue.
                    rule_to_extend_str_stack.pop()
                    continue 
                else:
                    print("--- extend this rule:", rule_to_extend_str)
                    rule_template = self.logic_template[head_predicate_idx][rule_to_extend_idx]
                    is_update_weight, is_continue, added_rule_str_list = self.add_one_predicate_to_existing_rule(head_predicate_idx, training_dataset, rule_template)
                    if not is_continue: #don't re-visit this rule.
                        rule_to_extend_str_stack.pop()
                    for added_rule_str in added_rule_str_list:
                        rule_to_extend_str_stack.append(added_rule_str)
            #if is_update_weight:
            #    self.generate_target( head_predicate_idx, testing_dataset, num_repeat=1)

        self.final_tune(head_predicate_idx, training_dataset)
        #self.generate_target( head_predicate_idx, testing_dataset, num_repeat=100)
        with open("./model/model-{}.pkl".format(tag),'wb') as f:
            pickle.dump(self, f)
        print("----- exit DFS -----", flush=1)

    def generate_target_one_sample(self, head_predicate_idx, data_sample, num_repeat):
        input_target_sample = {"time": data_sample[head_predicate_idx]["time"].copy(), "state": data_sample[head_predicate_idx]["state"].copy()}
        local_sample = copy.deepcopy(data_sample)
        if head_predicate_idx in self.instant_pred_set:
            s = np.array(input_target_sample["state"])
            t = np.array(input_target_sample["time"])
            idx = (s == 1)
            s = s[idx]
            t = t[idx]
            input_target_sample["time"] = list(t)
            input_target_sample["state"] = list(s)
            local_sample[head_predicate_idx]["time"] = list(t)
            local_sample[head_predicate_idx]["state"] = list(s)

        target_sample_length = len(local_sample[head_predicate_idx]["time"])
        
        if head_predicate_idx in self.survival_pred_set:
            # survival pred, calculate ACC
            
            local_sample_ID = 0
            local_data = {local_sample_ID: local_sample}
            integral = self.intensity_integral(head_predicate_idx, local_data, local_sample_ID)
            integral = integral.detach().numpy()
            survival_rate = np.exp(-integral)
            is_survival = int(survival_rate > 0.5)
            #print(is_survival)
            #print(input_target_sample["state"][-1])
            return [abs(is_survival - input_target_sample["state"][-1]),]

        else:
            

            # other preds, calculate MAE
            if target_sample_length <= 0:
                return []
                #return input_target_sample["time"]
            
            # obtain the maximal intensity
            T_max = local_sample[head_predicate_idx]["time"][-1] + self.Time_tolerance
            local_sample_ID = 0
            local_data = {local_sample_ID: local_sample}
            intensity_potential = []
            for t in np.arange(0, T_max, self.integral_resolution):
                t = t.item() #convert np scalar to float
                intensity = self.intensity(t, head_predicate_idx, local_data, local_sample_ID).detach()
                intensity_potential.append(intensity)
            if len(intensity_potential) == 0:
                return []
            intensity_max = max(intensity_potential)
            #print("intensity_max=", intensity_max,flush=1)

            generated_sample_time = []
            for i in range(target_sample_length):
                if i == 0:
                    T_min = 0
                else:
                    T_min = input_target_sample["time"][i-1]
                generated_time = list()
                local_sample[head_predicate_idx] = {"time": input_target_sample["time"][:i], "state":input_target_sample["state"][:i]} 
                for r in range(num_repeat):
                    t = T_min
                    #print("run idx=",r,flush=1)
                    while t < T_max:
                        time_to_event = np.random.exponential(scale=1.0/intensity_max).item()
                        t = t + time_to_event
                        #print("t=",t,flush=1)
                        if t >= T_max:
                            t=T_max
                            break
                        intensity = self.intensity(t, head_predicate_idx, local_data, local_sample_ID).detach()
                        ratio = min(intensity / intensity_max, 1)
                        flag = np.random.binomial(1, ratio)     # if flag = 1, accept, if flag = 0, regenerate
                        if flag-1 == 0: # accept
                            break
                    generated_time.append(t)
                t = sum(generated_time)/len(generated_time)
                generated_sample_time.append(t)
            
            gt = np.array(input_target_sample["time"])
            generated = np.array(generated_sample_time)
            ae = np.abs(gt- generated)
            #print("generated 1 sample, mae=",mae,flush=1)
            return ae

    def generate_target(self, head_predicate_idx, dataset, num_repeat=100):
        #print(self.instant_pred_set)
        print("----- start generation -----", flush=1)
        if self.debug_mode:
            num_repeat = 1
        arg_list = list()
        for sample_ID, data_sample in dataset.items():
            arg_list.append((head_predicate_idx, data_sample, num_repeat))
            
        cpu = cpu_count()
        worker_num = min(self.worker_num, cpu)
        worker_num = min(worker_num, len(arg_list))
        print("cpu num = {}, use {} workers, generate {} samples.".format(cpu, worker_num, len(arg_list)))
        with Timer("multiprocess generation") as t:
            if worker_num > 1: #multiprocessing
                with Pool(worker_num) as pool:
                    res = pool.starmap(self.generate_target_one_sample, arg_list)
            else: #single process, not use pool.
                res = [self.generate_target_one_sample(*arg) for arg in arg_list]
        res = np.concatenate(res)
        #print(res)
        if head_predicate_idx in self.survival_pred_set:
            res = 1 - res.mean()
            print("ACC =", res)
        else:
            res = res.mean()
            print("MAE =", res)
        print("----- exit generation -----", flush=1)

    def Brute(self, head_predicate_idx, dataset):
        print("----- start Brute -----", flush=1)
        self.start_time = time.time()
        head_predicate_sign = 1
        body_predicate_sign = 1
        
        # add len=1 rules
        for body_predicate_idx in self.body_pred_set: 
            for temporal_relation_type in [self.BEFORE, self.EQUAL]:
                rule = {
                    'body_predicate_idx':[body_predicate_idx],
                    'body_predicate_sign':[body_predicate_sign],
                    'head_predicate_sign':[head_predicate_sign], 
                    'temporal_relation_idx':[(body_predicate_idx, head_predicate_idx)], 
                    'temporal_relation_type' :[temporal_relation_type]
                    }
                self.logic_template[head_predicate_idx][self.num_formula] = rule 
                self.model_parameter[head_predicate_idx][self.num_formula]= {'weight':torch.autograd.Variable((torch.ones(1) * self.init_weight).double(), requires_grad=True)}
                self.num_formula += 1
        
        #
        max_length = 3
        for body_length in range(1, max_length):    
            rule_list = [template for idx,template in self.logic_template[head_predicate_idx].items() if len(template['body_predicate_idx']) == body_length]
            for rule in rule_list:
                for body_predicate_idx in self.body_pred_set: 
                    if body_predicate_idx in rule['body_predicate_idx']: 
                        continue
                    for temporal_relation_type in [self.BEFORE, self.EQUAL]:
                        for existing_predicate_idx in [head_predicate_idx] + rule['body_predicate_idx']:
                            new_rule = {
                                'body_predicate_idx':rule['body_predicate_idx']+[body_predicate_idx],
                                'body_predicate_sign':rule['body_predicate_sign']+[body_predicate_sign],
                                'head_predicate_sign':[head_predicate_sign], 
                                'temporal_relation_idx':rule['temporal_relation_idx']+[(body_predicate_idx, existing_predicate_idx)], 
                                'temporal_relation_type' :rule['temporal_relation_type']+[temporal_relation_type]
                                }
                            if self.check_repeat(new_rule, head_predicate_idx): # Repeated rule is not allowed.
                                continue
                            self.logic_template[head_predicate_idx][self.num_formula] = new_rule 
                            self.model_parameter[head_predicate_idx][self.num_formula]= {'weight':torch.autograd.Variable((torch.ones(1) * self.init_weight).double(), requires_grad=True)}
                            self.num_formula += 1
        
        with Timer("optimize log-likelihood (brute)") as t:
            l = self.optimize_log_likelihood_mp(head_predicate_idx, dataset)
            print("Log-likelihood (brute)= ", l, flush=1)

        print("----- exit Brute -----", flush=1)

                     



        







def set_rule(model, rule_set_str):
    import utils
    model_parameter, logic_template, head_predicate_idx, num_formula = utils.get_template(rule_set_str, model.predicate_notation)
    model.logic_template = logic_template
    model.model_parameter = model_parameter
    model.num_formula = num_formula

   
    
def generate(dataset_id, num_sample, rule_set_str, T_max=10):
    print("Start time is", datetime.datetime.now(),flush=1)
    #get data
    dataset =  get_data(dataset_id, num_sample)

    #get model
    model = get_model(dataset_id)

    #set initial rules if required
    if rule_set_str:
        set_rule(model, rule_set_str)
    model.print_rule()
    #generate
    model.generate_target(model.head_predicate_set[0], dataset, T_max)
    print("Finish time is", datetime.datetime.now())





def test_feature():
    dataset_path = './data/data-4.npy'
    print("datapath is ", dataset_path)
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    num_sample = 2560
    small_dataset = {i:dataset[i] for i in range(num_sample)}

    import generate_synthetic_data
    m, _ = generate_synthetic_data.get_logic_model_5()
    model = m.get_model()
    model.batch_size_grad = num_sample

    head_predicate_idx = 4
    
    #history = small_dataset[0]
    #template = model.logic_template[head_predicate_idx][2]
    def acc_test():
        for history in small_dataset.values():
            for template in model.logic_template[head_predicate_idx].values():
                for cur_time in range(1,10):
                    f = model.get_feature(cur_time, head_predicate_idx, history, template)
                    f_2 = model.get_feature_2(cur_time, head_predicate_idx, history, template)
                    if f.data[0] != f_2.data[0]:
                        print("f {}  !=  f_2 {}".format(f.data[0], f_2.data[0]))
    
    def time_test():
        with Timer("feature_2") as t:
            for history in small_dataset.values():
                for template in model.logic_template[head_predicate_idx].values():
                    for cur_time in range(1,10):
                        f = model.get_feature_2(cur_time, head_predicate_idx, history, template)
        #with Timer("feature_2") as t:
        #    for history in small_dataset.values():
        #        for template in model.logic_template[head_predicate_idx].values():
        #            for cur_time in range(1,10):
        #                f = model.get_feature(cur_time, head_predicate_idx, history, template)





if __name__ == "__main__":
    redirect_log_file()

    torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78
    
    







