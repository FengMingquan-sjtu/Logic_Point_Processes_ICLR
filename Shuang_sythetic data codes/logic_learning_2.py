import numpy as np
import itertools
import random
from multiprocessing import Pool, cpu_count
from collections import deque

import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import pickle
import cvxpy as cp

from generate_synthetic_data import Logic_Model_Generator
##################################################################

class Logic_Learning_Model(nn.Module):
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

        #tunable params
        self.Time_tolerance = 0.1
        self.integral_resolution = 0.3
        self.decay_rate = 1
        self.batch_size = 32
        self.num_batch_check_for_feature = 1
        self.num_batch_check_for_gradient = 20
        self.num_iter  = 5
        self.epsilon = 0.01
        self.threshold = 0.01
        self.learning_rate = 0.005
        self.max_rule_body_length = 3 #
        self.max_num_rule = 20
        self.batch_size_cp = 500 # batch size used in cp. If too large, may out of memory.
        self.batch_size_grad = 500 #batch_size used in optimize_log_grad.
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
        print("batch size = {}".format(self.batch_size))
        print("learning rate = {}".format(self.learning_rate))
        print("best_N = {}".format(self.best_N))
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
        


    def intensity(self, cur_time, head_predicate_idx, history):
        feature_formula = []
        weight_formula = []
        effect_formula = []

        for formula_idx in list(self.logic_template[head_predicate_idx].keys()):
            weight_formula.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])

            feature_formula.append(self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                    history=history, template=self.logic_template[head_predicate_idx][formula_idx]))
            effect_formula.append(self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                       history=history, template=self.logic_template[head_predicate_idx][formula_idx]))
        
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

    def get_feature(self, cur_time, head_predicate_idx, history, template):
        transition_time_dic = {}
        feature = torch.tensor([0], dtype=torch.float64)
        for idx, body_predicate_idx in enumerate(template['body_predicate_idx']):
            transition_time = np.array(history[body_predicate_idx]['time'])
            transition_state = np.array(history[body_predicate_idx]['state'])
            mask = (transition_time <= cur_time) * (transition_state == template['body_predicate_sign'][idx])
            transition_time_dic[body_predicate_idx] = transition_time[mask]
        transition_time_dic[head_predicate_idx] = [cur_time]
        ### get weights
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
                if template['temporal_relation_type'][idx] == 'EQUAL':
                    temporal_kernel *= (abs(time_difference) <= self.Time_tolerance) * np.exp(-self.decay_rate*(cur_time - time_combination_dic[temporal_relation_idx[0]]))
                if template['temporal_relation_type'][idx] == 'AFTER':
                    temporal_kernel *= (time_difference > self.Time_tolerance) * np.exp(-self.decay_rate*(cur_time - time_combination_dic[temporal_relation_idx[1]]))
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
            intensity_log_sum = self.intensity_log_sum(head_predicate_idx, data_sample)
            intensity_integral = self.intensity_integral(head_predicate_idx, data_sample, T_max)
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

    def intensity_log_sum(self, head_predicate_idx, data_sample):
        intensity_transition = []
        for t in data_sample[head_predicate_idx]['time'][:]:
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
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

    def intensity_integral(self, head_predicate_idx, data_sample, T_max):
        start_time = 0
        end_time = T_max
        intensity_grid = []
        for t in np.arange(start_time, end_time, self.integral_resolution):
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
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

    def optimize_log_likelihood(self, head_predicate_idx, dataset, T_max):
        print("---- start optimize_log_likelihood ----", flush=1)
        print("Rule set is:")
        self.print_rule()
        params = self.get_model_parameters(head_predicate_idx)
        optimizer = optim.Adam(params, lr=self.learning_rate)
        log_likelihood_batch = deque(list(), maxlen=self.num_batch_check_for_gradient)
        gradient_batch = deque(list(), maxlen=self.num_batch_check_for_gradient)
        params_batch = deque(list(), maxlen=self.num_batch_check_for_gradient)
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        epsilon = self.epsilon
        gradient_norm = 100

        for i in range(self.num_iter):
            sample_ID_list = list(dataset.keys())
            random.shuffle(sample_ID_list) #SGD
            for batch_idx in range(len(sample_ID_list)//self.batch_size):
                sample_ID_batch = sample_ID_list[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                log_likelihood = self.log_likelihood(head_predicate_idx, dataset, sample_ID_batch, T_max)
                l_1 = torch.sum(torch.abs(torch.stack(params)))
                loss = - log_likelihood + l_1
                loss.backward(retain_graph=True)
                optimizer.step()

                log_likelihood_batch.append(log_likelihood.data[0])

                batch_gradient = torch.autograd.grad(loss, params) # compute the batch gradient
                batch_gradient = torch.stack(batch_gradient).detach().numpy()
                #batch_gradient = np.min(np.abs(batch_gradient))
                gradient_batch.append(batch_gradient)
                gradient = np.mean(gradient_batch, axis=0)/self.batch_size # check the last N number of batch's gradient
                gradient_norm = np.linalg.norm(gradient)
                #gradient_norm = gradient
                params_detached = self.get_model_parameters(head_predicate_idx)
                params_detached = torch.stack(params_detached).detach().numpy()
                params_batch.append(params_detached)
                #print('Screening now, the moving avg batch gradient norm is', gradient_norm, flush=True)
                if len(gradient_batch) > self.num_batch_check_for_gradient and gradient_norm <= epsilon:
                    break
            if len(gradient_batch) > self.num_batch_check_for_gradient and gradient_norm <= epsilon:
                break

        #use the avg of last several batches log_likelihood
        log_likelihood = np.mean(log_likelihood_batch)/self.batch_size
        print('Finish optimize_log_likelihood, the log likelihood is', log_likelihood)
        param_array = np.mean(params_batch, axis=0).reshape(-1)
        
        self.set_model_parameters(head_predicate_idx, param_array)
        print("Params ", params)
        print("--------",flush=1)
        
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
    def intensity_integral_gradient(self, head_predicate_idx, data_sample, T_max):
        start_time = 0
        end_time = T_max
        intensity_gradient_grid = []
        for t in np.arange(start_time, end_time, self.integral_resolution):
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_gradient_grid.append(cur_intensity)   # due to that the derivative of exp(x) is still exp(x)
        integral_gradient_grid = torch.cat(intensity_gradient_grid, dim=0) * self.integral_resolution
        return integral_gradient_grid.detach() #detach, since multiprocessing needs requires_grad=False


    def log_likelihood_gradient(self, head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template):
        log_likelihood_gradient = torch.tensor([0], dtype=torch.float64)
        # iterate over samples
        sample_ID_batch = list(dataset.keys())
        if len(sample_ID_batch) < self.batch_size_grad:
            sample_ID_batch = random.sample(sample_ID_batch, self.batch_size_grad)
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

            log_likelihood_gradient += torch.sum(intensity_log_gradient[sample_ID] * new_feature_transition_times) - \
                                       torch.sum(intensity_integral_gradient_grid[sample_ID] * new_feature_grid_times, dim=0)
        return log_likelihood_gradient/ len(sample_ID_batch) # returns avg log-grad


    def optimize_log_likelihood_gradient(self, head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template):
        # in old codes, this function optimizes time relation params,
        # now there is no time relation params, so no optimization.
        gain = self.log_likelihood_gradient(head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template)
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
                feature_formula.append(self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                        history=data_sample, template=template))
                effect_formula.append(self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                        history=data_sample, template=template))
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
                    for temporal_relation_type in [self.BEFORE, self.EQUAL, self.AFTER]:
                        # create new rule
                        
                        
                        #temporally add new rule, to get likelihood.
                        self.logic_template[head_predicate_idx][self.num_formula] = {}
                        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = [body_predicate_idx]
                        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = [body_predicate_sign]
                        self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = [head_predicate_sign]
                        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = [(body_predicate_idx, head_predicate_idx)]
                        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = [temporal_relation_type]

                        self.model_parameter[head_predicate_idx][self.num_formula] = {}
                        self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable((torch.ones(1) ).double(), requires_grad=True)
                        self.model_parameter[head_predicate_idx][self.num_formula]['weight_cp'] = cp.Variable(1) 
                        self.num_formula +=1

                        # filter zero-feature rules.
                        feature_sum = self.get_feature_sum_for_screen(dataset, head_predicate_idx, template=self.logic_template[head_predicate_idx][self.num_formula-1])
                        if feature_sum == 0:
                            print("This rule is filtered, feature_sum=0, ", self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula-1], head_predicate_idx))
                            print("-------------",flush=1)
                        else:
                            #record the log-likelihood_gradient in performance gain
                            gain  = self.optimize_log_likelihood(head_predicate_idx, dataset, T_max)
                            print("Current rule is:", self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula-1], head_predicate_idx))
                            print("feature sum is", feature_sum)
                            print("log-likelihood is ", gain)
                            print("weight =", self.model_parameter[head_predicate_idx][self.num_formula-1]['weight'].item())
                            print("base =", self.model_parameter[head_predicate_idx]['base'].item())
                            print("----",flush=1)

                            #NOTE: Initialization does not require an accurate solution(CP).
                            #gain_cp = self.optimize_log_likelihood_cp(head_predicate_idx, dataset, T_max)
                            #print("log-likelihood-CP is ", gain_cp)
                            #print("weight=", self.model_parameter[head_predicate_idx][self.num_formula-1]['weight_cp'].value)
                            #print("base=", self.model_parameter[head_predicate_idx]['base_cp'].value)
                            #print("-------------",flush=1)
                            new_rule_table[head_predicate_idx]['performance_gain'].append(feature_sum)
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
                        
                        #Fast result for mimic.
                        if feature_sum !=0:
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
        l = self.optimize_log_likelihood(head_predicate_idx, dataset, T_max) #update base.
        print("Update Log-likelihood (torch) = ", l)

        if self.use_cp:
            l_cp = self.optimize_log_likelihood_cp(head_predicate_idx, dataset, T_max)
            print("Update Log-likelihood (cvxpy) = ", l_cp)

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
                        return True 
        return False


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
        print("start calculate intensity log and integral.", flush=1)
        sample_ID_batch = list(dataset.keys())
        intensity_log_gradient = dict()
        intensity_integral_gradient_grid = dict()
        for sample_ID in sample_ID_batch:
            data_sample = dataset[sample_ID]
            intensity_log_gradient[sample_ID] = self.intensity_log_gradient(head_predicate_idx, data_sample)
            intensity_integral_gradient_grid[sample_ID] = self.intensity_integral_gradient(head_predicate_idx, data_sample, T_max)

        ## search for the new rule from by minimizing the gradient of the log-likelihood
        arg_list = list()
        print("start enumerating candidate rules.", flush=1)
        for head_predicate_sign in [1, 0]:  # consider head_predicate_sign = 1/0
            for body_predicate_sign in [1, 0]: # consider head_predicate_sign = 1/0
                for body_predicate_idx in self.predicate_set:  
                    if body_predicate_idx == head_predicate_idx: # all the other predicates, excluding the head predicate, can be the potential body predicates
                        continue
                    for temporal_relation_type in [self.BEFORE, self.EQUAL, self.AFTER]:
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
                        
                        feature_sum = self.get_feature_sum_for_screen(dataset, head_predicate_idx, new_rule_template[head_predicate_idx])
                        if feature_sum == 0:
                            print("This rule is filtered, feature_sum={}, ".format(feature_sum), self.get_rule_str(new_rule_template[head_predicate_idx], head_predicate_idx))
                            print("-------------",flush=1)
                            continue

                        arg_list.append((head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template[head_predicate_idx]))

                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append(new_rule_template[head_predicate_idx]['body_predicate_idx'] )
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append(new_rule_template[head_predicate_idx]['body_predicate_sign'])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append(new_rule_template[head_predicate_idx]['head_predicate_sign'])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(new_rule_template[head_predicate_idx]['temporal_relation_idx'])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(new_rule_template[head_predicate_idx]['temporal_relation_type'])


        if len(arg_list) == 0: # No candidate rule generated.
            print("No candidate rule generated.")
            print("----- exit generate_rule_via_column_generation -----",flush=1)
            is_update_weight = False
            is_continue = False
            return is_update_weight, is_continue

        
        print("-------start multiprocess------",flush=1)
        cpu = cpu_count()
        worker_num = min(self.worker_num, cpu)
        print("cpu num = {}, use {} workers, process {} candidate rules.".format(cpu, worker_num, len(arg_list)))
        with Pool(worker_num) as pool:
            gain = pool.starmap(self.optimize_log_likelihood_gradient, arg_list)

        gain = np.array(gain)
        for i in range(len(gain)):
            rule_str = self.get_rule_str(arg_list[i][-1], head_predicate_idx)
            rule_gain = gain[i]
            print("log-likelihood-grad = {:.5f}, Rule = {}".format(rule_gain, rule_str))
            print("-------------")


        print("------Select N best rule-------")
        # choose the N-best rules that lead to the optimal log-likelihood
        idx_gain = sorted(list(enumerate(gain)), key=lambda x:x[1], reverse=True) # sort by gain, descending
        is_update_weight = False
        is_continue = False
        #print("idx_gain", idx_gain)
        for i in range(self.best_N):
            if i >= len(idx_gain):
                break
            idx, best_gain = idx_gain[i]
        
            if  best_gain > self.threshold:
                # add new rule
                self.logic_template[head_predicate_idx][self.num_formula] = {}
                self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

                print("Best rule is:", self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula], head_predicate_idx))
                print("Best log-likelihood-grad =", best_gain)
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
                print("best gain {} does not meet thershold {}.".format(best_gain, self.threshold))
                break

        if is_update_weight:
            # update model parameter
            l = self.optimize_log_likelihood(head_predicate_idx, dataset, T_max)
            print("Update Log-likelihood (torch)= ", l, flush=1)

            if self.use_cp:
                l_cp = self.optimize_log_likelihood_cp(head_predicate_idx, dataset, T_max)
                print("Update Log-likelihood (cvxpy)= ", l_cp)

            #Copy cvxpy to weight
            #for f_idx in range(0, self.num_formula+1):
            #    w = self.model_parameter[head_predicate_idx][f_idx]['weight_cp'].value[0]
            #    self.model_parameter[head_predicate_idx][f_idx]['weight'] = torch.autograd.Variable((torch.ones(1) * w).double(), requires_grad=True)
            #self.model_parameter[head_predicate_idx]['base'] = torch.autograd.Variable((torch.ones(1) * self.model_parameter[head_predicate_idx]['base_cp'].value[0]).double(), requires_grad=True)
                
        print("----- exit generate_rule_via_column_generation -----",flush=1)
        return is_update_weight, is_continue
            


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
        sample_ID_batch = list(dataset.keys())
        intensity_log_gradient = dict()
        intensity_integral_gradient_grid = dict()
        for sample_ID in sample_ID_batch:
            data_sample = dataset[sample_ID]
            intensity_log_gradient[sample_ID] = self.intensity_log_gradient(head_predicate_idx, data_sample)
            intensity_integral_gradient_grid[sample_ID] = self.intensity_integral_gradient(head_predicate_idx, data_sample, T_max)

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
                for temporal_relation_type in [self.BEFORE, self.EQUAL, self.AFTER]:
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
                        
                        feature_sum = self.get_feature_sum_for_screen(dataset, head_predicate_idx, new_rule_template[head_predicate_idx])
                        if feature_sum == 0:
                            print("This rule is filtered, feature_sum={}, ".format(feature_sum), self.get_rule_str(new_rule_template[head_predicate_idx], head_predicate_idx))
                            print("-------------",flush=1)
                            continue

                        arg_list.append((head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template[head_predicate_idx]))

                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append(new_rule_template[head_predicate_idx]['body_predicate_idx'] )
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append(new_rule_template[head_predicate_idx]['body_predicate_sign'])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append(new_rule_template[head_predicate_idx]['head_predicate_sign'])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(new_rule_template[head_predicate_idx]['temporal_relation_idx'])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(new_rule_template[head_predicate_idx]['temporal_relation_type'])
   

        if len(arg_list) == 0: # No candidate rule generated.
            print("No candidate rule generated.")
            print("----- exit add_one_predicate_to_existing_rule  -----",flush=1)
            is_update_weight = False
            is_continue = False
            return is_update_weight, is_continue

        print("-------start multiprocess------",flush=1)
        cpu = cpu_count()
        worker_num = min(self.worker_num, cpu)
        print("cpu num = {}, use {} workers, process {} candidate rules.".format(cpu, worker_num, len(arg_list)))
        with Pool(worker_num) as pool:
            gain = pool.starmap(self.optimize_log_likelihood_gradient, arg_list)

        gain = np.array(gain)
        for i in range(len(gain)):
            rule_str = self.get_rule_str(arg_list[i][-1], head_predicate_idx)
            rule_gain = gain[i]
            print("log-likelihood-grad = {:.5f}, Rule = {}".format(rule_gain, rule_str))
            print("-------------")


        print("------Select N best rule-------")
        # choose the N-best rules that lead to the optimal log-likelihood
        idx_gain = sorted(list(enumerate(gain)), key=lambda x:x[1], reverse=True) # sort by gain, descending
        is_update_weight = False
        is_continue = False
        for i in range(self.best_N):
            if i >= len(idx_gain):
                break
            idx, best_gain = idx_gain[-i]
        
            if  best_gain > self.threshold:
                # add new rule
                self.logic_template[head_predicate_idx][self.num_formula] = {}
                self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
                self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

                print("Best rule is:", self.get_rule_str(self.logic_template[head_predicate_idx][self.num_formula], head_predicate_idx))
                print("Best log-likelihood-grad =", best_gain)
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
                print("best gain {} does not meet thershold {}.".format(best_gain, self.threshold))
                break

        if is_update_weight:
            # update model parameter
            l = self.optimize_log_likelihood(head_predicate_idx, dataset, T_max)
            print("Update Log-likelihood (torch)= ", l, flush=1)

            if self.use_cp:
                l_cp = self.optimize_log_likelihood_cp(head_predicate_idx, dataset, T_max)
                print("Update Log-likelihood (cvxpy)= ", l_cp)

            #Copy cvxpy to weight
            #for f_idx in range(0, self.num_formula+1):
            #    w = self.model_parameter[head_predicate_idx][f_idx]['weight_cp'].value[0]
            #    self.model_parameter[head_predicate_idx][f_idx]['weight'] = torch.autograd.Variable((torch.ones(1) * w).double(), requires_grad=True)
            #self.model_parameter[head_predicate_idx]['base'] = torch.autograd.Variable((torch.ones(1) * self.model_parameter[head_predicate_idx]['base_cp'].value[0]).double(), requires_grad=True)
                
        print("----- exit add_one_predicate_to_existing_rule -----",flush=1)
        return is_update_weight, is_continue


    def prune_rules_with_small_weights(self):
        # if num_formula -=1, then how to move existing formulas?
        # maybe only prune when learning finishes.
        # TODO
        pass


    def search_algorithm(self, head_predicate_idx, dataset, T_max):
        self.print_info()
        self.initialize_rule_set(head_predicate_idx, dataset, T_max)
        print("Initialize with this rule:")
        self.print_rule_cp()
        #Begin Breadth(width) First Search
        #generate new rule from scratch
        while self.num_formula < self.max_num_rule:
            is_update_weight, is_continue = self.generate_rule_via_column_generation(head_predicate_idx, dataset, T_max)
            
            if is_update_weight:
                print("Added simple rules. Current rule set is:")
                self.print_rule_cp()
            if not is_continue:
                break
        
        #generate new rule by extending existing rules
        for cur_body_length in range(1, self.max_rule_body_length + 1):
            if self.num_formula >= self.max_num_rule:
                print("Maximum rule number reached.")
                break
            for existing_rule_template in list(self.logic_template[head_predicate_idx].values()):
                if self.num_formula >= self.max_num_rule:
                    print("Maximum rule number reached.")
                    break
                if len(existing_rule_template['body_predicate_idx']) == cur_body_length: 
                    is_update_weight, is_continue = self.add_one_predicate_to_existing_rule(head_predicate_idx, dataset, T_max, existing_rule_template)
                    if is_update_weight:
                        print("Extended an existing rule. Current rule set is:")
                        self.print_rule_cp()

        print("Train finished, Final rule set is:")
        self.print_rule()
        #self.prune_rule_by_small_weights()

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

                
                




if __name__ == "__main__":
    head_predicate_idx = [4]
    model = Logic_Learning_Model(head_predicate_idx = head_predicate_idx)
    model.predicate_set= [0, 1, 2, 3, 4] # the set of all meaningful predicates
    model.predicate_notation = ['A', 'B', 'C', 'D', 'E']
    T_max = 10
    dataset = np.load('data.npy', allow_pickle='TRUE').item()
    num_sample = 20000 #dataset size
    print("dataset size is {}".format(num_sample) )
    

    small_dataset = {i:dataset[i] for i in range(num_sample)}
    model.batch_size_cp = num_sample  # sample used by cp
    model.batch_size_grad = 2000
    

    model.search_algorithm(head_predicate_idx[0], small_dataset, T_max)

    with open("model.pkl",'wb') as f:
        pickle.dump(model, f)







