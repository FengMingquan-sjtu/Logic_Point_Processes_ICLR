
import itertools
import random
from multiprocessing import Pool, cpu_count
from collections import deque
import time
import os
import sys
import datetime
import warnings

import numpy as np
import scipy
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import pickle
import cvxpy as cp


#random.seed(1)

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[%s] " % self.name, end="")
        dt = time.time() - self.tstart
        if dt < 60:
            print("Elapsed: {:.4f} sec.".format(dt))
        elif dt < 3600:
            print("Elapsed: {:.4f} min.".format(dt / 60))
        elif dt < 86400:
            print("Elapsed: {:.4f} hour.".format(dt / 3600))
        else:
            print("Elapsed: {:.4f} day.".format(dt / 86400))

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
        self.head_predicate_set = head_predicate_idx.copy()  # the index set of only one head predicates

        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.num_formula = 0
        self.feature_cache = dict()
        self.deleted_rules = set()
        self.low_grad_rules = dict()

        # tunable params
        self.time_window = 10
        self.Time_tolerance = 0.1
        self.integral_resolution = 0.1
        self.decay_rate = 1
        self.batch_size = 64
        self.num_batch_check_for_feature = 1
        self.num_batch_check_for_gradient = 20
        self.num_batch_no_update_limit_opt = 300
        self.num_batch_no_update_limit_ucb = 4
        self.num_iter  = 5
        self.epsilon = 0.003
        self.gain_threshold = 0.02
        self.low_grad_threshold = 0.01
        self.low_grad_tolerance = 2
        self.weight_threshold = 0.01
        self.strict_weight_threshold = 0.1
        self.learning_rate = 0.005
        self.max_rule_body_length = 3 #
        self.max_num_rule = 30
        self.batch_size_cp = 500 # batch size used in cp. If too large, may out of memory.
        self.batch_size_grad = 500 #batch_size used in optimize_log_grad.
        self.batch_size_init_ucb = 5
        self.explore_rule_num_ucb = 8
        self.explore_batch_size_ucb = 500
        self.use_cp = False
        self.worker_num = 8
        self.best_N = 2
        
        #claim parameters and rule set
        self.model_parameter = {}
        self.logic_template = {}

        for idx in self.head_predicate_set:
            self.model_parameter[idx] = {}
            self.model_parameter[idx]['base'] = torch.autograd.Variable((torch.ones(1) * -0.2).double(), requires_grad=True)
            self.model_parameter[idx]['base_cp'] = cp.Variable(1)
            self.logic_template[idx] = {}

    def print_info(self):
        print("-----key model information----")
        for valuename, value in vars(self).items():
            if isinstance(value, float) or isinstance(value, int):
                print("{}={}".format(valuename, value))
        print("----",flush=1)

    def get_model_parameters(self, head_predicate_idx):
        # collect all parameters in a list, used as input of Adam optimizer.
        parameters = list()
        parameters.append(self.model_parameter[head_predicate_idx]['base'])
        for formula_idx in range(self.num_formula): #TODO:potential bug
            parameters.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
        return parameters
    
    def set_model_parameters(self, head_predicate_idx, param_array):
        # set model params
        parameters = list()
        base = param_array[0]
        self.model_parameter[head_predicate_idx]['base'] = torch.autograd.Variable((torch.ones(1) * base).double(), requires_grad=True)
        for formula_idx in range(self.num_formula):
            weight = param_array[formula_idx+1]
            self.model_parameter[head_predicate_idx][formula_idx]['weight'] = torch.autograd.Variable((torch.ones(1) * weight).double(), requires_grad=True)
        
    def delete_rules(self, head_predicate_idx, formula_idx_list):
        # delete formulas listed in formula_idx_list
        #TODO: only consider single head

        # delete feature cache, and add rule_str to self.deleted_rules:
        for formula_idx in formula_idx_list:
            rule_str = self.get_rule_str(self.logic_template[head_predicate_idx][formula_idx], head_predicate_idx)
            if rule_str in self.feature_cache:
                self.feature_cache[rule_str] = dict()
            self.deleted_rules.add(rule_str)

        # delete weight and logic-template
        tmp_logic_template = dict()
        tmp_model_parameter = dict()
        tmp_model_parameter['base'] = self.model_parameter[head_predicate_idx]['base']
        tmp_model_parameter['base_cp'] = self.model_parameter[head_predicate_idx]['base_cp']
        
        new_formula_idx = 0
        for formula_idx in range(self.num_formula):
            if not formula_idx in formula_idx_list:
                tmp_logic_template[new_formula_idx] = self.logic_template[head_predicate_idx][formula_idx]
                tmp_model_parameter[new_formula_idx] = dict()
                tmp_model_parameter[new_formula_idx]["weight"] = self.model_parameter[head_predicate_idx][formula_idx]['weight']
                tmp_model_parameter[new_formula_idx]["weight_cp"] = self.model_parameter[head_predicate_idx][formula_idx]['weight_cp']
                new_formula_idx += 1

        self.logic_template[head_predicate_idx] = tmp_logic_template
        self.model_parameter[head_predicate_idx] = tmp_model_parameter
        self.num_formula -= len(formula_idx_list)



    def intensity(self, cur_time, head_predicate_idx, dataset, sample_ID, use_cache=True):
        feature_formula = []
        weight_formula = []
        effect_formula = []

        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            weight_formula.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
            if use_cache:
                f = self.get_feature_with_cache(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                    dataset=dataset, sample_ID=sample_ID, template=self.logic_template[head_predicate_idx][formula_idx])
            else:
                f = self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx, history= dataset[sample_ID], template=self.logic_template[head_predicate_idx][formula_idx])
            feature_formula.append(f)
            effect_formula.append(self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                       history=dataset[sample_ID], template=self.logic_template[head_predicate_idx][formula_idx]))
        
        if len(weight_formula)>0:
            #intensity = torch.exp(torch.cat(weight_formula, dim=0))/torch.sum(torch.exp(torch.cat(weight_formula, dim=0)), dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)
            #NOTE: Softmax on weight leads to error when len(weight) = 1. Gradient on weight is very small.
            intensity =  torch.cat(weight_formula, dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)

        else:
            intensity = torch.zeros(1)
        intensity = self.model_parameter[head_predicate_idx]['base'] + torch.sum(intensity)
        intensity = torch.exp(intensity)

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

    def get_feature_with_cache(self, cur_time, head_predicate_idx, dataset, sample_ID, template):
        # feature cache is a nested dict: 
        # key1 = rule_str
        # key2 = (sample_ID,cur_time)
        # self.feature_cache[key1][key2] = feature
        # ::split 2 keys to reduce memory storage.
        feature = self.get_feature(cur_time, head_predicate_idx, dataset[sample_ID], template)
        return feature
        warnings.warn("disable all cache")
        key1 = self.get_rule_str(template, head_predicate_idx)
        if not key1 in self.feature_cache:
            self.feature_cache[key1] = dict()
        
        key2 = (sample_ID, cur_time)
        #print("key1 is",key1)
        #print("key2 is",key2)
        if key2 in self.feature_cache[key1]:
            feature = self.feature_cache[key1][key2]
        else:
            feature = self.get_feature(cur_time, head_predicate_idx, dataset[sample_ID], template)
            self.feature_cache[key1][key2] = feature 
        return feature

    def get_feature(self, cur_time, head_predicate_idx, history, template):
        transition_time_dic = {}
        feature = torch.tensor([0], dtype=torch.float64)
        for idx, body_predicate_idx in enumerate(template['body_predicate_idx']):
            transition_time = np.array(history[body_predicate_idx]['time'])
            transition_state = np.array(history[body_predicate_idx]['state'])
            
            if [body_predicate_idx, head_predicate_idx] in template['temporal_relation_idx']:
                #for time-relation with target, filter events by time-relation
                temporal_idx = template['temporal_relation_idx'].index([body_predicate_idx, head_predicate_idx])
                if template['temporal_relation_type'][temporal_idx] == self.BEFORE:
                    mask = (transition_time >= cur_time - self.time_window) * (transition_time <= cur_time - self.Time_tolerance) * (transition_state == template['body_predicate_sign'][idx])
                elif template['temporal_relation_type'][temporal_idx] == self.EQUAL:
                    mask = (transition_time >= cur_time - self.Time_tolerance) * (transition_time <= cur_time) * (transition_state == template['body_predicate_sign'][idx])
                else:
                    raise ValueError
            else:
                mask = (transition_time >= cur_time-self.time_window) * (transition_time <= cur_time) * (transition_state == template['body_predicate_sign'][idx])

            transition_time_dic[body_predicate_idx] = transition_time[mask]
        transition_time_dic[head_predicate_idx] = [cur_time]
        ### get weights
        #print(transition_time_dic)
        # compute features whenever any item of the transition_item_dic is nonempty
        history_transition_len = [len(i) for i in transition_time_dic.values()]
        if min(history_transition_len) > 0:
            # need to compute feature using logic rules
            time_combination = np.array(list(itertools.product(*transition_time_dic.values())))
            time_combination_dic = {}
            for i, idx in enumerate(list(transition_time_dic.keys())):
                time_combination_dic[idx] = time_combination[:, i]
            temporal_kernel = np.ones(len(time_combination))
            for idx, temporal_relation_idx in enumerate(template['temporal_relation_idx']):
                time_difference = time_combination_dic[temporal_relation_idx[0]] - time_combination_dic[temporal_relation_idx[1]]
                if template['temporal_relation_type'][idx] == 'BEFORE':
                    temporal_kernel *= (time_difference < - self.Time_tolerance) * np.exp(-self.decay_rate *(cur_time - time_combination_dic[temporal_relation_idx[0]]))
                elif template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.Time_tolerance) * np.exp(-self.decay_rate*(cur_time - time_combination_dic[temporal_relation_idx[0]]))
            feature = torch.tensor([np.sum(temporal_kernel)], dtype=torch.float64)
        return feature

    def get_formula_effect(self, cur_time, head_predicate_idx, history, template):
        ## Note this part is very important!! For generator, this should be np.sum(cur_time > head_transition_time) - 1
        ## Since at the transition times, choose the intensity function right before the transition time
        head_transition_time = np.array(history[head_predicate_idx]['time'])
        head_transition_state = np.array(history[head_predicate_idx]['state'])
        if len(head_transition_time) == 0:
            cur_state = 0
            counter_state = 1 - cur_state
        else:
            idx = np.sum(cur_time > head_transition_time) - 1
            cur_state = head_transition_state[idx]
            counter_state = 1 - cur_state
        if counter_state == template['head_predicate_sign']:
            formula_effect = torch.tensor([1], dtype=torch.float64)
        else:
            formula_effect = torch.tensor([-1], dtype=torch.float64)
        return formula_effect

    ### the following functions are for optimizing the logic weights
    def log_likelihood(self, head_predicate_idx, dataset, sample_ID_batch, T_max):
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        # iterate over samples
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            intensity_log_sum = self.intensity_log_sum(head_predicate_idx, dataset, sample_ID)
            intensity_integral = self.intensity_integral(head_predicate_idx, dataset, sample_ID, T_max)
            log_likelihood += (intensity_log_sum - intensity_integral)
        return log_likelihood
    
    def log_likelihood_cp(self, head_predicate_idx, dataset, sample_ID_batch, T_max):
        log_likelihood = np.zeros(1)
        for sample_ID in sample_ID_batch:
            data_sample = dataset[sample_ID]
            intensity_log_sum = self.intensity_log_sum_cp(head_predicate_idx, data_sample)
            intensity_integral = self.intensity_integral_cp(head_predicate_idx, data_sample, T_max)
            log_likelihood += intensity_log_sum - intensity_integral
        return log_likelihood

    def intensity_log_sum(self, head_predicate_idx, dataset, sample_ID):
        intensity_transition = []
        for t in dataset[sample_ID][head_predicate_idx]['time'][:]:
            cur_intensity = self.intensity(t, head_predicate_idx, dataset, sample_ID)
            intensity_transition.append(cur_intensity)
        if len(intensity_transition) == 0: # only survival term, not event happens
            log_sum = torch.tensor([0], dtype=torch.float64)
        else:
            log_sum = torch.sum(torch.log(torch.cat(intensity_transition, dim=0)))
        return log_sum
    
    def intensity_log_sum_cp(self, head_predicate_idx, data_sample):
        log_sum = np.zeros(1)
        for t in data_sample[head_predicate_idx]['time'][:]:
            # directly calculate log-intensity, to avoid DCPError of cvxpy.
            log_intensity = self.intensity_cp(t, head_predicate_idx, data_sample, mapping=False)
            log_sum += log_intensity
        return log_sum

    def intensity_integral(self, head_predicate_idx, dataset, sample_ID, T_max):
        start_time = 0
        end_time = T_max
        intensity_grid = []
        for t in np.arange(start_time, end_time, self.integral_resolution):
            cur_intensity = self.intensity(t, head_predicate_idx, dataset, sample_ID)
            intensity_grid.append(cur_intensity)
        integral = torch.sum(torch.cat(intensity_grid, dim=0) * self.integral_resolution)
        return integral

    def intensity_integral_cp(self, head_predicate_idx, data_sample, T_max):
        start_time = 0
        end_time = T_max
        integral = np.zeros(1)
        for t in np.arange(start_time, end_time, self.integral_resolution):
            cur_intensity = self.intensity_cp(t, head_predicate_idx, data_sample)
            integral += cur_intensity * self.integral_resolution
        return integral

    def optimize_log_likelihood(self, head_predicate_idx, dataset, T_max, verbose=True):
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
                    log_likelihood = self.log_likelihood(head_predicate_idx, dataset, sample_ID_batch, T_max)
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

    def _optimize_log_likelihood_mp_worker(self, optimizer, head_predicate_idx, dataset, sample_ID_batch, T_max):
        # update weitghs
        optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
        log_likelihood = self.log_likelihood(head_predicate_idx, dataset, sample_ID_batch, T_max)
        params = self.get_model_parameters(head_predicate_idx)
        l_1 = torch.sum(torch.abs(torch.stack(params)))
        loss = - log_likelihood + l_1
        loss.backward()
        optimizer.step()

        params = torch.stack(params).detach().numpy()
        log_likelihood = log_likelihood.detach().numpy()

        return log_likelihood, params

        #return some values
        #log_likelihood = log_likelihood.data[0]/len(sample_ID_batch)
        #batch_gradient = torch.autograd.grad(loss, params) # compute the batch gradient
        #batch_gradient = torch.stack(batch_gradient).detach().numpy()/len(sample_ID_batch)
        #params = torch.stack(params).detach().numpy()

        #return log_likelihood, batch_gradient, params


    def optimize_log_likelihood_mp(self, head_predicate_idx, dataset, T_max, verbose=True):
        print("---- start optimize_log_likelihood multi-process----", flush=1)
        print("Rule set is:")
        self.print_rule()
        worker_num = min(self.worker_num, cpu_count())
        params = self.get_model_parameters(head_predicate_idx)
        optimizer = optim.Adam(params, lr=self.learning_rate)
        batch_num = len(dataset.keys())// self.batch_size
        arg_list = list()
        for i in range(self.num_iter):
            sample_ID_list = list(dataset.keys())
            random.shuffle(sample_ID_list) #random dataset order
            
            for batch_idx in range(batch_num):
                sample_ID_batch = sample_ID_list[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
                args = (optimizer, head_predicate_idx, dataset, sample_ID_batch, T_max)
                arg_list.append(args)

        worker_num = min(worker_num, len(arg_list))
        print("use {} workers for {} batches".format(worker_num, len(arg_list)))
        with Pool(worker_num) as p:
            ret = p.starmap(self._optimize_log_likelihood_mp_worker, arg_list)
  
        log_likelihood_list, params_list = zip(*ret) 
        log_likelihood = np.mean(log_likelihood_list[-self.num_batch_check_for_gradient:])/self.batch_size
        param_array = np.mean(params_list[-self.num_batch_check_for_gradient:], axis=0).reshape(-1)
        self.set_model_parameters(head_predicate_idx, param_array)
        print("optimized rule weights are:")
        self.print_rule()
        print("---- exit optimize_log_likelihood multi-process----", flush=1)
        
        return log_likelihood

    def optimize_log_likelihood_cp(self, head_predicate_idx, dataset, T_max):
        # optimize using cvxpy
        print("start optimize using cp:", flush=1)
        sample_ID_batch =  random.sample(dataset.keys(), self.batch_size_cp)
        log_likelihood = self.log_likelihood_cp(head_predicate_idx, dataset, sample_ID_batch, T_max)
        objective = cp.Maximize(log_likelihood)
        prob = cp.Problem(objective)
        
        opt_log_likelihood = prob.solve(verbose=False, solver="SCS")

        return opt_log_likelihood / len(sample_ID_batch)


    def intensity_log_gradient(self, head_predicate_idx, data_sample):
        # intensity_transition = []
        # for t in data_sample[head_predicate_idx]['time'][1:]:
        #     cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
        #     intensity_transition.append(cur_intensity)
        # if len(intensity_transition) == 0:  # only survival term, not event happens
        #     log_gradient = torch.tensor([0], dtype=torch.float64)
        # else:
        #     log_gradient = torch.cat(intensity_transition, dim=0).pow(-1)
        # return log_gradient
        return torch.tensor([1.0],dtype=torch.float64) #intensity_log_gradient of exp kernel is always 1.

    ### the following functions are to compute sub-problem objective function
    def intensity_integral_gradient(self, head_predicate_idx, dataset, sample_ID, T_max):
        start_time = 0
        end_time = T_max
        intensity_gradient_grid = []
        for t in np.arange(start_time, end_time, self.integral_resolution):
            cur_intensity = self.intensity(t, head_predicate_idx, dataset, sample_ID)
            cur_intensity = cur_intensity.detach() #detach, since multiprocessing needs requires_grad=False
            intensity_gradient_grid.append(cur_intensity)   # due to that the derivative of exp(x) is still exp(x)
        integral_gradient_grid = torch.cat(intensity_gradient_grid, dim=0) * self.integral_resolution
        return integral_gradient_grid


    def log_likelihood_gradient(self, head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template, batch_size=0):
        #calculate gradient of likelihood at w_new = 0.

        log_likelihood_grad_list = list()
        # iterate over samples
        sample_ID_batch = list(dataset.keys())
        if batch_size == 0:
            batch_size = sample_ID_batch
        elif len(sample_ID_batch) > batch_size:
            print("Random select {} samples from {} samples".format(batch_size, len(sample_ID_batch)), flush=1)
            sample_ID_batch = random.sample(sample_ID_batch, batch_size)
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            # compute the log_intensity_gradient, integral_gradient_grid using existing rules

            start_time = 0
            end_time = T_max # Note for different sample_ID, the T_max can be different
            # compute new feature at the transition times
            new_feature_transition_times = []
            for t in data_sample[head_predicate_idx]['time'][:]:
                f = self.get_feature(cur_time=t, head_predicate_idx=head_predicate_idx, history=data_sample, template =new_rule_template)
                # NOTE: log-like-grad does not need cache, use multi-processing instead.
                #f = self.get_feature_with_cache(cur_time=t, head_predicate_idx=head_predicate_idx, dataset=dataset, sample_ID=sample_ID, template =new_rule_template)
                fe = self.get_formula_effect(cur_time=t, head_predicate_idx=head_predicate_idx, history=data_sample, template =new_rule_template)
                new_feature_transition_times.append(f * fe)
            new_feature_grid_times = []
            for t in np.arange(start_time, end_time, self.integral_resolution):
                 
                f = self.get_feature(cur_time=t, head_predicate_idx=head_predicate_idx, history=data_sample, template =new_rule_template)
                # NOTE: log-like-grad does not need cache, use multi-processing instead.
                #f = self.get_feature_with_cache(cur_time=t, head_predicate_idx=head_predicate_idx, dataset=dataset, sample_ID=sample_ID, template =new_rule_template)
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

            log_likelihood_grad = torch.sum(intensity_log_gradient[sample_ID] * new_feature_transition_times) - \
                                       torch.sum(intensity_integral_gradient_grid[sample_ID] * new_feature_grid_times, dim=0)
            log_likelihood_grad_list.append(log_likelihood_grad)
        mean_grad = np.mean(log_likelihood_grad_list) 
        #note: we assume l1 penalty gradient is zero, at w=0
        return mean_grad, log_likelihood_grad_list # returns avg log-grad adn list.


    def optimize_log_likelihood_gradient(self, head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template, batch_size=0):
        # in old codes, this function optimizes time relation params,
        # now there is no time relation params, so no optimization.
        gain = self.log_likelihood_gradient(head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template, batch_size)
        return gain



    # here we use the  width-first search to add body predicates
    # we assume that the number of important rules should be smaller thant K
    # we assume that the length of the body predicates should be smaller thant L
    def get_feature_sum_for_screen(self, dataset, head_predicate_idx, template):
        sample_ID_batch = random.sample(dataset.keys(), self.batch_size * self.num_batch_check_for_feature)
        #sample_ID_batch = dataset.keys()
        feature_sum = 0
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            effect_formula = []
            feature_formula= []

            for cur_time in data_sample[head_predicate_idx]['time'][:]:
                feature_formula.append(self.get_featurewith_cache(cur_time=t, head_predicate_idx=head_predicate_idx, dataset=dataset, sample_ID=sample_ID, template =new_rule_template))
                effect_formula.append(self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,history=data_sample, template=template))
            if len(feature_formula) != 0:
                feature_sum += torch.sum(torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0))
                #print("data_sample=", data_sample)
                #print("effect_formula =", torch.cat(effect_formula, dim=0))
                #print("feature_formula=", torch.cat(feature_formula, dim=0))
        return feature_sum
                
            
    def initialize_rule_set(self, head_predicate_idx, dataset, T_max):
        print("----- start initialize_rule_set -----")

        new_rule_table = {}
        new_rule_table[head_predicate_idx] = {}
        new_rule_table[head_predicate_idx]['body_predicate_idx'] = []
        new_rule_table[head_predicate_idx]['body_predicate_sign'] = []  # use 1 to indicate True; use 0 to indicate False
        new_rule_table[head_predicate_idx]['head_predicate_sign'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_idx'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_type'] = []
        new_rule_table[head_predicate_idx]['performance_gain'] = []
        new_rule_table[head_predicate_idx]['weight'] = []
        new_rule_table[head_predicate_idx]['weight_cp'] = []

        print("start enumerating candicate rules")
        ## search for the new rule from by minimizing the gradient of the log-likelihood
        flag = 0
        for head_predicate_sign in [1, 0]:  # consider head_predicate_sign = 1/0
            for body_predicate_sign in [1, 0]:
                for body_predicate_idx in self.predicate_set:  
                    if body_predicate_idx == head_predicate_idx: # all the other predicates, excluding the head predicate, can be the potential body predicates
                        continue
                    #NOTE: due to bug#36, remove self.AFTER in enumeration
                    for temporal_relation_type in [self.BEFORE, self.EQUAL]:
                        # create new rule

                        #temporally add new rule, to get likelihood.
                        self.logic_template[head_predicate_idx][self.num_formula] = {}
                        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = [body_predicate_idx]
                        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = [body_predicate_sign]
                        self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = [head_predicate_sign]
                        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = [(body_predicate_idx, head_predicate_idx)]
                        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = [temporal_relation_type]

                        self.model_parameter[head_predicate_idx][self.num_formula] = {}
                        self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable((torch.ones(1)*0.01).double(), requires_grad=True)
                        self.model_parameter[head_predicate_idx][self.num_formula]['weight_cp'] = cp.Variable(1) 
                        self.num_formula +=1

                        #NOTE: due to bug#36, remove self.AFTER in enumeration, thus feature sum filter is useless.
                        # filter zero-feature rules.
                        # feature_sum = self.get_feature_sum_for_screen(dataset, head_predicate_idx, template=self.logic_template[head_predicate_idx][self.num_formula-1])
                        # if feature_sum == 0:
                        #     print("This rule is filtered, feature_sum=0, ", self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula-1], head_predicate_idx))
                        #     print("-------------",flush=1)
                        
                        #record the log-likelihood_gradient in performance gain
                        with Timer("optimize log-likelihood") as t:
                            #self.optimize_log_likelihood( head_predicate_idx, dataset, T_max)
                            gain  = self.optimize_log_likelihood_mp(head_predicate_idx, dataset, T_max)
                        print("Current rule is:", self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula-1], head_predicate_idx))
                        #print("feature sum is", feature_sum)
                        print("log-likelihood is ", gain)
                        print("weight =", self.model_parameter[head_predicate_idx][self.num_formula-1]['weight'].item())
                        print("base =", self.model_parameter[head_predicate_idx]['base'].item())
                        print("----",flush=1)

                        #NOTE: Initialization does not require an accurate solution(CP).
                        
                        new_rule_table[head_predicate_idx]['performance_gain'].append(gain)
                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append([body_predicate_idx])
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append([body_predicate_sign])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append([head_predicate_sign])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append([(body_predicate_idx, head_predicate_idx)])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append([temporal_relation_type])
                        new_rule_table[head_predicate_idx]['weight'].append(self.model_parameter[head_predicate_idx][self.num_formula-1]['weight'])
                        new_rule_table[head_predicate_idx]['weight_cp'].append(self.model_parameter[head_predicate_idx][self.num_formula-1]['weight_cp'])

                        #remove the new rule.
                        self.num_formula -=1
                        self.logic_template[head_predicate_idx][self.num_formula] = {}
                        self.model_parameter[head_predicate_idx][self.num_formula] = {}
                        
                        #Fast result for large dataset like mimic.
                        print("NOTE: Random initialization for fast result.")
                        flag = 1
                        break
                    if flag:
                        break
                if flag:
                    break
            if flag:
                break

        print("------Select best rule-------")
        idx = np.argmax(new_rule_table[head_predicate_idx]['performance_gain'])
        best_gain = new_rule_table[head_predicate_idx]['performance_gain'][idx]

        # add new rule
        self.logic_template[head_predicate_idx][self.num_formula] = {}
        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

        print("Best initial rule is:", self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula], head_predicate_idx))
        print("Best log-likelihood =", best_gain)
        # add model parameter
        self.model_parameter[head_predicate_idx][self.num_formula] = {}
        self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = new_rule_table[head_predicate_idx]['weight'][idx]
        self.model_parameter[head_predicate_idx][self.num_formula]['weight_cp'] = new_rule_table[head_predicate_idx]['weight_cp'][idx]
        self.num_formula += 1

        #update params
        print("NOTE: Random initialization for fast result.")
        #with Timer("optimize log-likelihood") as t:
        #    l = self.optimize_log_likelihood_mp(head_predicate_idx, dataset, T_max) #update base.
        #print("Update Log-likelihood (torch) = ", l)

        #if self.use_cp:
        #    l_cp = self.optimize_log_likelihood_cp(head_predicate_idx, dataset, T_max)
        #    print("Update Log-likelihood (cvxpy) = ", l_cp)

        #Copy CVXPY to weight
        #w = self.model_parameter[head_predicate_idx][self.num_formula-1]['weight_cp'].value[0]
        #self.model_parameter[head_predicate_idx][self.num_formula-1]['weight'] = torch.autograd.Variable((torch.ones(1) * w).double(), requires_grad=True)
        #self.model_parameter[head_predicate_idx]['base'] = torch.autograd.Variable((torch.ones(1) * self.model_parameter[head_predicate_idx]['base_cp'].value[0]).double(), requires_grad=True)

        print("----- exit initialize_rule_set -----",flush=1)
        return


    def check_repeat(self, new_rule, head_predicate_idx):
        new_rule_body_predicate_set = set(zip(new_rule['body_predicate_idx'], new_rule['body_predicate_sign']))
        new_rule_temporal_relation_set = set(zip(new_rule['temporal_relation_idx'], new_rule['temporal_relation_type']))
        for rule in self.logic_template[head_predicate_idx].values():
            if rule['head_predicate_sign'] == new_rule['head_predicate_sign']:
                body_predicate_set = set(zip(rule['body_predicate_idx'], rule['body_predicate_sign']))
                if body_predicate_set == new_rule_body_predicate_set:
                    temporal_relation_set = set(zip(rule['temporal_relation_idx'], rule['temporal_relation_type']))
                    if temporal_relation_set == new_rule_temporal_relation_set:
                        return True #repeat with existing rules
        rule_str = self.get_rule_str(new_rule, head_predicate_idx)
        if rule_str in self.deleted_rules:
            return True #repeat with deleted rules
        return False #not repeat


    def generate_rule_via_column_generation(self, head_predicate_idx, dataset, T_max):
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
        intensity_log_gradient, intensity_integral_gradient_grid = self.get_intensity_and_integral_grad( head_predicate_idx, dataset, T_max)
        ## search for the new rule from by minimizing the gradient of the log-likelihood
        arg_list = list()
        print("start enumerating candidate rules.", flush=1)
        
        for body_predicate_sign in [1, 0]: # consider head_predicate_sign = 1/0
            for body_predicate_idx in self.predicate_set:  
                if body_predicate_idx == head_predicate_idx: # all the other predicates, excluding the head predicate, can be the potential body predicates
                    continue
                for temporal_relation_type in [self.BEFORE, self.EQUAL]:
                    # create new rule
                    head_predicate_sign = 1   # due to bug#73
                    new_rule_template = {}
                    new_rule_template[head_predicate_idx]= {}
                    new_rule_template[head_predicate_idx]['body_predicate_idx'] = [body_predicate_idx]
                    new_rule_template[head_predicate_idx]['body_predicate_sign'] = [body_predicate_sign]  # use 1 to indicate True; use 0 to indicate False
                    new_rule_template[head_predicate_idx]['head_predicate_sign'] = [head_predicate_sign]
                    new_rule_template[head_predicate_idx]['temporal_relation_idx'] = [(body_predicate_idx, head_predicate_idx)]
                    new_rule_template[head_predicate_idx]['temporal_relation_type'] = [temporal_relation_type]

                    if self.check_repeat(new_rule_template[head_predicate_idx], head_predicate_idx): # Repeated rule is not allowed.
                        continue
            
                    arg_list.append((head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template[head_predicate_idx]))

                    new_rule_table[head_predicate_idx]['body_predicate_idx'].append(new_rule_template[head_predicate_idx]['body_predicate_idx'] )
                    new_rule_table[head_predicate_idx]['body_predicate_sign'].append(new_rule_template[head_predicate_idx]['body_predicate_sign'])
                    new_rule_table[head_predicate_idx]['head_predicate_sign'].append(new_rule_template[head_predicate_idx]['head_predicate_sign'])
                    new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(new_rule_template[head_predicate_idx]['temporal_relation_idx'])
                    new_rule_table[head_predicate_idx]['temporal_relation_type'].append(new_rule_template[head_predicate_idx]['temporal_relation_type'])


        is_update_weight, is_continue = self.select_and_add_new_rule(head_predicate_idx, arg_list, new_rule_table, dataset, T_max)
        
        print("----- exit generate_rule_via_column_generation -----",flush=1)
        return is_update_weight, is_continue
            
    def get_intensity_and_integral_grad(self, head_predicate_idx, dataset, T_max):
        print("---start calculate intensity grad and integral grad.---", flush=1)
        with Timer("calculate intensity grad and integral grad") as t:
            sample_ID_batch = list(dataset.keys())
            intensity_log_gradient = dict()
            intensity_integral_gradient_grid = dict()
            arg_list = list()  #args for parallel intensity_integral_gradient_grid()
            for sample_ID in sample_ID_batch:
                intensity_log_gradient[sample_ID] = self.intensity_log_gradient(head_predicate_idx, data_sample = dataset[sample_ID])
                arg_list.append((head_predicate_idx, dataset, sample_ID, T_max)) #args for parallel intensity_integral_gradient_grid()
            
            worker_num = min(self.worker_num, cpu_count())
            worker_num = min(worker_num, len(arg_list))
            with Pool(worker_num) as p:
                integral_grad_list = p.starmap(self.intensity_integral_gradient, arg_list) 

            for idx, sample_ID in enumerate(sample_ID_batch):
                intensity_integral_gradient_grid[sample_ID] = integral_grad_list[idx]
        print("---exit calculate intensity grad and integral grad.---", flush=1)
        return intensity_log_gradient, intensity_integral_gradient_grid


    def add_one_predicate_to_existing_rule(self, head_predicate_idx, dataset, T_max, existing_rule_template):
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
        intensity_log_gradient, intensity_integral_gradient_grid = self.get_intensity_and_integral_grad( head_predicate_idx, dataset, T_max)



        ## search for the new rule from by minimizing the gradient of the log-likelihood
        #be careful, do NOT modify existing rule.
        arg_list = list()
        print("start enumerating candidate rules.", flush=1)
        existing_predicate_idx_list = [head_predicate_idx] + existing_rule_template['body_predicate_idx']
        for body_predicate_sign in [1, 0]:
            for body_predicate_idx in self.predicate_set:
                if body_predicate_idx in existing_predicate_idx_list: 
                    # these predicates are not allowed.
                    continue 
                #NOTE: due to bug#36, remove self.AFTER in enumeration
                for temporal_relation_type in [self.BEFORE, self.EQUAL]:
                    for existing_predicate_idx in existing_predicate_idx_list:
                        
                        # create new rule
                        new_rule_template = {}
                        new_rule_template[head_predicate_idx]= {}
                        new_rule_template[head_predicate_idx]['body_predicate_idx'] = [body_predicate_idx] + existing_rule_template['body_predicate_idx']
                        new_rule_template[head_predicate_idx]['body_predicate_sign'] = [body_predicate_sign] + existing_rule_template['body_predicate_sign']
                        new_rule_template[head_predicate_idx]['head_predicate_sign'] = [] + existing_rule_template['head_predicate_sign']
                        new_rule_template[head_predicate_idx]['temporal_relation_idx'] = [(body_predicate_idx, existing_predicate_idx)] + existing_rule_template['temporal_relation_idx']
                        new_rule_template[head_predicate_idx]['temporal_relation_type'] = [temporal_relation_type] + existing_rule_template['temporal_relation_type']

                        if self.check_repeat(new_rule_template[head_predicate_idx], head_predicate_idx): # Repeated rule is not allowed.
                            continue
                        

                        arg_list.append((head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template[head_predicate_idx]))

                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append(new_rule_template[head_predicate_idx]['body_predicate_idx'] )
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append(new_rule_template[head_predicate_idx]['body_predicate_sign'])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append(new_rule_template[head_predicate_idx]['head_predicate_sign'])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(new_rule_template[head_predicate_idx]['temporal_relation_idx'])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(new_rule_template[head_predicate_idx]['temporal_relation_type'])
   

        is_update_weight, is_continue = self.select_and_add_new_rule(head_predicate_idx, arg_list, new_rule_table, dataset, T_max)

        print("----- exit add_one_predicate_to_existing_rule -----",flush=1)
        return is_update_weight, is_continue

    def _update_ucb_dict(self, gain_ucb_dict, idx, gain_list):
        gain_ucb_dict["gain_list"][idx].extend(gain_list)
        std_gain = np.std(gain_ucb_dict["gain_list"][idx], ddof=1) #ddof=1 provides an unbiased estimator of the variance
        mean_gain = np.mean(gain_ucb_dict["gain_list"][idx])
        bound_gain = std_gain + mean_gain
        gain_ucb_dict["std"][idx] = std_gain
        gain_ucb_dict["mean"][idx] = mean_gain
        gain_ucb_dict["bound"][idx] = bound_gain

    def __ucb(self):
        return
        # note: this function is ucb part of select_and_add_new_rule()
        # ucb is more efficient when #candidate and #samples are large.
        print("-------start ucb ------",flush=1)
        cpu = cpu_count()
        worker_num = min(self.worker_num, cpu)
        with Timer("UCB") as t:
            
            #initialization
            gain_ucb_dict = dict()
            gain_ucb_dict["gain_list"] = [list() for i in range(len(arg_list))]
            gain_ucb_dict["mean"] = [0] * len(arg_list)
            gain_ucb_dict["std"] = [0] * len(arg_list)
            gain_ucb_dict["bound"] = [0] * len(arg_list)
            

            #ucb main loop
            num_batch_no_update = 0
            cur_batch = 0
            best_N = min(self.best_N, len(arg_list))
            best_idx_set = set(range(best_N))
            for batch_idx in range(len(dataset.keys())//self.explore_batch_size_ucb):
                #print("UCB: {}th batch".format(cur_batch))
                
                if batch_idx > 0:
                    explore_rule_num = min(self.explore_rule_num_ucb, len(arg_list))
                    sorted_idx = sorted(list(range(len(arg_list))), key=lambda x:gain_ucb_dict["mean"][x], reverse=True) # sort by mean, descending
                    explore_idx_list = sorted_idx[:explore_rule_num]
                else:
                    explore_idx_list = list(range(len(arg_list))) #explore all candidate for initialization
                explore_arg_list = list()
                # each time, only explore a batch of whole data.
                sample_ID_list = list(dataset.keys())[batch_idx*self.explore_batch_size_ucb : (batch_idx+1)*self.explore_batch_size_ucb]
                batch_dataset = {sample_ID:dataset[sample_ID] for sample_ID in sample_ID_list}
                for idx in explore_idx_list:
                    # replace the whole dataset with batch_dataset
                    head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template = arg_list[idx]
                    arg = (head_predicate_idx, batch_dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template, self.explore_batch_size_ucb)
                    explore_arg_list.append(arg)
                
                #use multi-processing to calculate grad (TODO:maybe slow)
                tmp = self.feature_cache
                self.feature_cache = dict() #to fix bug#39, clear feature cache, avoid copying cache in multiprocessing (extremely slow)
                worker_num_ = min(worker_num, len(explore_arg_list)) 
                with Pool(worker_num_) as pool:
                    gain_explore = pool.starmap(self.optimize_log_likelihood_gradient, explore_arg_list)
                self.feature_cache = tmp #recover cache
                
                # update ucb dict
                for i, explore_idx in enumerate(explore_idx_list):
                    gain_list = gain_explore[i][1]
                    self._update_ucb_dict(gain_ucb_dict, explore_idx, gain_list)

                for explore_idx in explore_idx_list:
                    rule_str = self.get_rule_str(arg_list[explore_idx][-1], head_predicate_idx)
                    mean, std, bound = gain_ucb_dict["mean"][explore_idx], gain_ucb_dict["std"][explore_idx], gain_ucb_dict["bound"][explore_idx]
                    print("Explore update:  log-likelihood-grad(ucb): mean= {:.5f}, std= {:.5f}, bound= {:.5f}, Rule = {}".format(mean, std, bound, rule_str))

                # update best idx
                sorted_idx = sorted(list(range(len(arg_list))), key=lambda x:gain_ucb_dict["mean"][x], reverse=True) # sort by mean, descending
                cur_best_idx_set = set(sorted_idx[:best_N])
                if best_idx_set != cur_best_idx_set:
                    best_idx_set = cur_best_idx_set
                    num_batch_no_update = 0
                else:
                    num_batch_no_update +=1
                    if num_batch_no_update >= self.num_batch_no_update_limit_ucb:
                        break

                cur_batch +=1
                print("----------")
        
        
        print("UCB ends after {} batches".format(cur_batch) )

        for best_idx in best_idx_set:
            rule_str = self.get_rule_str(arg_list[best_idx][-1], head_predicate_idx)
            mean, std, bound = gain_ucb_dict["mean"][best_idx], gain_ucb_dict["std"][best_idx], gain_ucb_dict["bound"][best_idx]
            print("Exploit final decision:  log-likelihood-grad(ucb): mean= {:.5f}, std= {:.5f}, bound= {:.5f}, Rule = {}".format(mean, std, bound, rule_str))

        print("-------end ucb ------",flush=1)

    def select_and_add_new_rule(self, head_predicate_idx, arg_list, new_rule_table, dataset, T_max):
        
        print("----- start select_and_add_new_rule -----",flush=1)
        if len(arg_list) == 0: # No candidate rule generated.
            print("No candidate rule generated.")
            is_update_weight = False
            is_continue = False
            print("----- exit select_and_add_new_rule -----",flush=1)
            return is_update_weight, is_continue

        # all-data gradient
        print("-------start multiprocess------",flush=1)
        self.batch_size_grad = len(dataset.keys()) # run all samples for grad.
        cpu = cpu_count()
        worker_num = min(self.worker_num, cpu)
        worker_num = min(worker_num, len(arg_list))
        print("cpu num = {}, use {} workers, process {} candidate rules.".format(cpu, worker_num, len(arg_list)))
        with Timer("multiprocess log-grad") as t:
            
            if worker_num > 1: #multiprocessing
                tmp = self.feature_cache
                self.feature_cache = dict() #to fix bug#39, clear feature cache, avoid copying cache in multiprocessing (extremely slow)
                with Pool(worker_num) as pool:
                    gain_all_data = pool.starmap(self.optimize_log_likelihood_gradient, arg_list)
                self.feature_cache = tmp #recover cache
            else: #single process, not use pool.
                gain_all_data = [self.optimize_log_likelihood_gradient(*arg) for arg in arg_list]
        mean_gain_all_data, gain_list_all_data  = list(zip(*gain_all_data))
        mean_gain_all_data = list(mean_gain_all_data)
        print("-------end multiprocess------",flush=1)
        

        for idx, gain_ in enumerate(mean_gain_all_data):
            if gain_ < 0:
                # if gain<0, revert head-pred sign and gain. using symmetricity.
                sign = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx][0]
                new_rule_table[head_predicate_idx]['head_predicate_sign'][idx][0] = 1 - sign
                arg_list[idx][-1]['head_predicate_sign'][0] = 1 - sign
                mean_gain_all_data[idx] = - mean_gain_all_data[idx]

                
        #delete low gain candidate rules
        for idx, gain_ in enumerate(mean_gain_all_data):
            if gain_ < self.low_grad_threshold:
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
        sorted_idx_gain_all_data = sorted(list(enumerate(mean_gain_all_data)), key=lambda x:x[1], reverse=True) # sort by gain, descending
        for idx, gain in sorted_idx_gain_all_data:
            rule_str = self.get_rule_str(arg_list[idx][-1], head_predicate_idx)
            std = np.std(gain_list_all_data[idx],ddof=1) #ddof=1 for unbiased estimation
            print("log-likelihood-grad(all-data) mean= {:.5f}, std={:.5f}, Rule = {}".format(gain, std, rule_str))
            print("-------------", flush=True)
        is_update_weight = False
        is_continue = False
        for i in range(self.best_N):
            if i >= len(sorted_idx_gain_all_data):
                break
            idx, best_gain = sorted_idx_gain_all_data[i]
        
            if  best_gain > self.gain_threshold:
                # add new rule
                self.logic_template[head_predicate_idx][self.num_formula] = {}
                self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

                print("Best rule is:", self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula], head_predicate_idx))
                print("Best log-likelihood-grad(all-data) =", best_gain)
                # add model parameter
                self.model_parameter[head_predicate_idx][self.num_formula] = {}
                self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)
                self.model_parameter[head_predicate_idx][self.num_formula]['weight_cp'] = cp.Variable(1)
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
            with Timer("optimize log-likelihood") as t:
                l = self.optimize_log_likelihood_mp(head_predicate_idx, dataset, T_max)
            print("Update Log-likelihood (torch)= ", l, flush=1)

            if self.use_cp:
                l_cp = self.optimize_log_likelihood_cp(head_predicate_idx, dataset, T_max)
                print("Update Log-likelihood (cvxpy)= ", l_cp)

            print("Added rule and re-fitted weights. Current rule set is:")
            self.print_rule_cp()
            
            # prune after added.
            is_strict = self.num_formula >= self.max_num_rule #if too many rules, use strict threshold, otherwise, use normal threshold.
            is_pruned = self.prune_rules_with_small_weights(head_predicate_idx, dataset, T_max, is_strict)
            if is_pruned: #after prunning, maybe add more rules.
                is_continue = True

        print("----- exit select_and_add_new_rule -----",flush=1)
        return is_update_weight, is_continue


    def prune_rules_with_small_weights(self, head_predicate_idx, dataset, T_max, is_strict=False):
        formula_idx_list = list()
        if is_strict:
            thershold = self.strict_weight_threshold
        else:
            thershold = self.weight_threshold

        for formula_idx in range(self.num_formula):
            w = self.model_parameter[head_predicate_idx][formula_idx]['weight'].detach()
            if w < thershold:
                formula_idx_list.append(formula_idx)
        if len(formula_idx_list) > 0:
            print("delete these rules:",formula_idx_list)
            self.delete_rules(head_predicate_idx, formula_idx_list)
            #refit weights
            print("start re-fit weights", flush=1)
            with Timer("optimize log-likelihood") as t:
                l = self.optimize_log_likelihood_mp(head_predicate_idx, dataset, T_max)
            print("update Log-likelihood (torch)= ", l, flush=1)
            print("Deleted some rules and refited weights, Current rule set is:")
            self.print_rule_cp()
            return True
        return False
            

    def search_algorithm(self, head_predicate_idx, dataset, T_max):
        print("----- start search_algorithm -----", flush=1)
        self.print_info()
        #self.initialize_rule_set(head_predicate_idx, dataset, T_max)
        #print("Initialize with this rule:")
        #self.print_rule_cp()
        #Begin Breadth(width) First Search
        #generate new rule from scratch
        is_continue = True
        while self.num_formula < self.max_num_rule and is_continue:
            is_update_weight, is_continue = self.generate_rule_via_column_generation(head_predicate_idx, dataset, T_max)
            
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

                if template_to_extend is None:
                    flag = False
                    break
                
                rule_str = self.get_rule_str(template_to_extend, head_predicate_idx)
                extended_rules.add(rule_str)
                print("start to extend this rule:", rule_str)
                
                #extend the selected rule.
                is_continue = True
                while(self.num_formula < self.max_num_rule and is_continue):
                    is_update_weight, is_continue = self.add_one_predicate_to_existing_rule(head_predicate_idx, dataset, T_max, template_to_extend)


        print("Train finished, rule set is:")
        self.print_rule()

        # final prune, with strict threshold of weight.
        pruned = True
        while pruned:
            pruned = self.prune_rules_with_small_weights(head_predicate_idx, dataset, T_max, is_strict=True )
        
        # final optimize, with large iter_num
        self.iter_num = 50
        self.optimize_log_likelihood_mp(head_predicate_idx, dataset, T_max)
        print("Train finished, rule set is:")
        self.print_rule()
        print("----- exit search_algorithm -----", flush=1)

    def print_rule(self):
        for head_predicate_idx, rules in self.logic_template.items():
            print("Head = {}, base = {:.4f}".format(self.predicate_notation[head_predicate_idx], self.model_parameter[head_predicate_idx]['base'].data[0]))
            for rule_id, rule in rules.items():
                rule_str = "Rule{}: ".format(rule_id)
                rule_str += self.get_rule_str(rule, head_predicate_idx)
                weight = self.model_parameter[head_predicate_idx][rule_id]['weight'].data[0]
                rule_str += ", weight={:.4f}".format(weight)
                print(rule_str)
    
    def print_rule_cp(self):
        for head_predicate_idx, rules in self.logic_template.items():
            base = self.model_parameter[head_predicate_idx]['base'].item()
            base_cp = self.model_parameter[head_predicate_idx]['base_cp'].value
            if base_cp:
                print("Head = {}, base(torch) = {:.4f}, base(cp) = {:.4f},".format(self.predicate_notation[head_predicate_idx], base, base_cp[0]))
            else:
                print("Head = {}, base(torch) = {:.4f},".format(self.predicate_notation[head_predicate_idx], base))
            for rule_id, rule in rules.items():
                rule_str = "Rule{}: ".format(rule_id)
                rule_str += self.get_rule_str(rule, head_predicate_idx)
                weight = self.model_parameter[head_predicate_idx][rule_id]['weight'].item()
                weight_cp = self.model_parameter[head_predicate_idx][rule_id]['weight_cp'].value
                if weight_cp:
                    rule_str += ", weight(torch)={:.4f}, weight(cp)={:.4f}.".format(weight, weight_cp[0])
                else:
                    rule_str += ", weight(torch)={:.4f}.".format(weight)
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

                

def fit_1():
    print("Start time is", datetime.datetime.now(),flush=1)
    head_predicate_idx = [4]
    model = Logic_Learning_Model(head_predicate_idx = head_predicate_idx)
    model.predicate_set= [0, 1, 2, 3, 4] # the set of all meaningful predicates
    model.predicate_notation = ['A', 'B', 'C', 'D', 'E']
    T_max = 10
    dataset_path = './data/data-1.npy'
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    num_sample = 2000 #dataset size
    print("dataset path is ", dataset_path)
    print("dataset size is {}".format(num_sample) )

    small_dataset = {i:dataset[i] for i in range(num_sample)}
    model.batch_size_cp = num_sample  # sample used by cp
    model.batch_size_grad = num_sample
    
    with Timer("search_algorithm") as t:
        model.search_algorithm(head_predicate_idx[0], small_dataset, T_max)

    print("Finish time is", datetime.datetime.now())

    if not os.path.exists("./model"):
        os.makedirs("./model")

    with open("./model/model-1.pkl",'wb') as f:
        pickle.dump(model, f)            

def fit_2():
    print("Start time is", datetime.datetime.now(),flush=1)
    head_predicate_idx = [5]
    model = Logic_Learning_Model(head_predicate_idx = head_predicate_idx)
    model.predicate_set= [0, 1, 2, 3, 4, 5] # the set of all meaningful predicates
    model.predicate_notation = ['A', 'B', 'C', 'D', 'E', 'F']
    T_max = 10
    dataset_path = './data/data-2.npy'
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    num_sample =1000 #dataset size
    print("dataset path is ", dataset_path)
    print("dataset size is {}".format(num_sample) )

    small_dataset = {i:dataset[i] for i in range(num_sample)}
    model.batch_size_grad = num_sample
    #model.num_iter = 1
    #warnings.warn("!!using very small num_iter !!")

    with Timer("search_algorithm") as t:
        model.search_algorithm(head_predicate_idx[0], small_dataset, T_max)

    print("Finish time is", datetime.datetime.now())
    if not os.path.exists("./model"):
        os.makedirs("./model")
    with open("./model/model-2.pkl",'wb') as f:
        pickle.dump(model, f)       

def fit_4():
    print("Start time is", datetime.datetime.now(),flush=1)
    head_predicate_idx = [4]
    model = Logic_Learning_Model(head_predicate_idx = head_predicate_idx)
    model.predicate_set= [0, 1, 2, 3] # the set of all meaningful predicates
    model.predicate_notation = ['A', 'B', 'C', 'D', 'E']
    T_max = 10
    dataset_path = './data/data-4.npy'
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    num_sample =1000 #dataset size
    print("dataset path is ", dataset_path)
    print("dataset size is {}".format(num_sample))

    small_dataset = {i:dataset[i] for i in range(num_sample)}
    model.batch_size_grad = num_sample

    with Timer("search_algorithm") as t:
        model.search_algorithm(head_predicate_idx[0], small_dataset, T_max)

    print("Finish time is", datetime.datetime.now())
    if not os.path.exists("./model"):
        os.makedirs("./model")
    with open("./model/model-4.pkl",'wb') as f:
        pickle.dump(model, f)       

def fit_5():
    print("Start time is", datetime.datetime.now(),flush=1)
    head_predicate_idx = [4]
    model = Logic_Learning_Model(head_predicate_idx = head_predicate_idx)
    model.predicate_set= [0, 1, 2, 3] # the set of all meaningful predicates
    model.predicate_notation = ['A', 'B', 'C', 'D', 'E']
    T_max = 10
    dataset_path = './data/data-5.npy'
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    num_sample =1000 #dataset size
    print("dataset path is ", dataset_path)
    print("dataset size is {}".format(num_sample))

    small_dataset = {i:dataset[i] for i in range(num_sample)}
    model.batch_size_grad = num_sample
    #model.num_iter = 1
    #warnings.warn("!!using very small num_iter !!")

    with Timer("search_algorithm") as t:
        model.search_algorithm(head_predicate_idx[0], small_dataset, T_max)

    print("Finish time is", datetime.datetime.now())
    if not os.path.exists("./model"):
        os.makedirs("./model")
    with open("./model/model-4.pkl",'wb') as f:
        pickle.dump(model, f)       

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

def fit(dataset_id, num_sample):
    print("Start time is", datetime.datetime.now(),flush=1)

    #get model
    from generate_synthetic_data import get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5
    logic_model_funcs = [None,get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5]
    m, _ = logic_model_funcs[dataset_id]()
    model = m.get_model_for_learn()

    #get data
    dataset_path = './data/data-{}.npy'.format(dataset_id)
    print("dataset_path is ",dataset_path)
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    if len(dataset.keys())> num_sample: 
        dataset = {i:dataset[i] for i in range(num_sample)}
    num_sample = len(dataset.keys())
    print("sample num is ", num_sample)
    model.batch_size_grad = num_sample #use all sample for grad

    with Timer("search_algorithm") as t:
        model.search_algorithm(model.head_predicate_set[0], dataset, T_max=10)
    
    print("Finish time is", datetime.datetime.now())
    if not os.path.exists("./model"):
        os.makedirs("./model")
    with open("./model/model-{}.pkl".format(dataset_id),'wb') as f:
        pickle.dump(model, f)       





def test_feature():
    dataset_path = './data/data-5.npy'
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    num_sample =100 
    small_dataset = {i:dataset[i] for i in range(num_sample)}

    import generate_synthetic_data
    m, _ = generate_synthetic_data.get_logic_model_5()
    model = m.get_model()
    model.batch_size_grad = num_sample

    head_predicate_idx = 4
    cur_time = 7
    history = small_dataset[0]
    template = model.logic_template[head_predicate_idx][2]
    #print(history)
    #print(template)
    f = model.get_feature(cur_time, head_predicate_idx, history, template)
    print(f)

if __name__ == "__main__":
    redirect_log_file()
    fit(4,1280)
    #test_feature()








