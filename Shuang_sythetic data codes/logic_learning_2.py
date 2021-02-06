import numpy as np
import itertools
import random
from generate_synthetic_data import Logic_Model_Generator
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import pickle
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

        self.predicate_set= [0, 1, 2, 3, 4] # the set of all meaningful predicates
        self.predicate_notation = ['A','B','C','D','E']
        self.head_predicate_set = head_predicate_idx.copy()  # the index set of only one head predicates

        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.num_formula = 0

        #tunable params
        self.Time_tolerance = 0.1
        self.integral_resolution = 0.3
        self.decay_rate = 1
        self.batch_size = 16
        self.num_batch_check_for_gradient = 20
        self.num_iter  = 5
        self.max_rule_body_length = 3 #
        self.max_num_rule = 5
        
        
        #claim parameters and rule set
        self.model_parameter = {}
        self.logic_template = {}

        for idx in head_predicate_idx:
            self.model_parameter[idx] = {}
            self.model_parameter[idx]['base'] = torch.autograd.Variable((torch.ones(1) * -0.2).double(), requires_grad=True)
            self.logic_template[idx] = {}

    def get_model_parameters(self):
        # collect all parameters in a list, used as input of Adam optimizer.
        parameters = list()
        for head_predicate_idx in self.head_predicate_set:
            parameters.append(self.model_parameter[head_predicate_idx]['base'])
            for formula_idx in self.model_parameter[head_predicate_idx].keys():
                if formula_idx == 'base':
                    continue
                parameters.append(self.model_parameter[head_predicate_idx][formula_idx]['weight'])
        return parameters


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
            
            # without Softmax on weight, gradient is like:
            #batch_gradient =  (tensor([-12.2186], dtype=torch.float64), tensor([-0.9893], dtype=torch.float64), tensor([4.3846], dtype=torch.float64))
            # with Softmax, gradient is like:
            #batch_gradient =  (tensor([13.8510], dtype=torch.float64), tensor([-9.9920e-16], dtype=torch.float64), tensor([7.7664], dtype=torch.float64))
        else:
            intensity = torch.zeros(1)
        intensity = self.model_parameter[head_predicate_idx]['base'] + torch.sum(intensity)
        intensity = torch.exp(intensity)

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
    def log_likelihood(self, dataset, sample_ID_batch, T_max):
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        # iterate over samples
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            for head_predicate_idx in self.head_predicate_set:
                intensity_log_sum = self.intensity_log_sum(head_predicate_idx, data_sample)
                intensity_integral = self.intensity_integral(head_predicate_idx, data_sample, T_max)
                log_likelihood += (intensity_log_sum - intensity_integral)
        return log_likelihood

    def intensity_log_sum(self, head_predicate_idx, data_sample):
        intensity_transition = []
        for t in data_sample[head_predicate_idx]['time'][1:]:
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_transition.append(cur_intensity)
        if len(intensity_transition) == 0: # only survival term, not event happens
            log_sum = torch.tensor([0], dtype=torch.float64)
        else:
            log_sum = torch.sum(torch.log(torch.cat(intensity_transition, dim=0)))
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


    def optimize_log_likelihood(self, dataset, T_max):
        print("start optimize:", flush=1)
        self.print_rule()
        params = self.get_model_parameters()
        optimizer = optim.Adam(params, lr=0.1)
        log_likelihood_batch = []
        gradient_batch = []
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        epsilon = 0.1
        gradient_norm = 100

        for i in range(self.num_iter):
            sample_ID_list = list(dataset.keys())
            random.shuffle(sample_ID_list) #SGD
            for batch_idx in range(len(sample_ID_list)//self.batch_size):
                sample_ID_batch = sample_ID_list[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                loss = -self.log_likelihood(dataset, sample_ID_batch, T_max)
                #print(loss)
                loss.backward(retain_graph=True)
                optimizer.step()

                log_likelihood_batch.append(- loss.data[0])
                batch_gradient = torch.autograd.grad(loss, params) # compute the batch gradient
                print("batch_gradient = ", batch_gradient)
                
                batch_gradient = torch.stack(batch_gradient).detach().numpy()
                #batch_gradient = np.min(np.abs(batch_gradient))
                gradient_batch.append(batch_gradient)
                gradient = np.mean(gradient_batch[-self.num_batch_check_for_gradient:], axis=0)/self.batch_size # check the last N number of batch's gradient
                gradient_norm = np.linalg.norm(gradient)
                #gradient_norm = gradient
                print('Screening now, the moving avg batch gradient norm is', gradient_norm, flush=True)
                if gradient_norm <= epsilon:
                    break
            if gradient_norm <= epsilon:
                    break

        log_likelihood = np.mean(log_likelihood_batch[-self.num_batch_check_for_gradient:])/self.batch_size
        print('Finish screening one variable, the log likelihood is', log_likelihood)
        print("Params ", params)
        #use the last batch log_likelihood
        return log_likelihood



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
        return integral_gradient_grid


    def log_likelihood_gradient(self, head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template):
        log_likelihood_gradient = torch.tensor([0], dtype=torch.float64)
        # iterate over samples
        sample_ID_batch = list(dataset.keys())
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            # compute the log_intensity_gradient, integral_gradient_grid using existing rules

            start_time = 0
            end_time = T_max # Note for different sample_ID, the T_max can be different
            # compute new feature at the transition times
            new_feature_transition_times = []
            for t in data_sample[head_predicate_idx]['time'][1:]:
                new_feature_transition_times.append(self.get_feature(cur_time=t, head_predicate_idx=head_predicate_idx,
                                                    history=data_sample, template =new_rule_template))
            new_feature_grid_times = []
            for t in np.arange(start_time, end_time, self.integral_resolution):
                new_feature_grid_times.append(self.get_feature(cur_time=t, head_predicate_idx=head_predicate_idx,
                                                    history=data_sample, template =new_rule_template))
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
        return log_likelihood_gradient


    def optimize_log_likelihood_gradient(self, head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template):
        # in old codes, this function optimizes time relation params,
        # now there is no time relation params, so no optimization.
        gain = self.log_likelihood_gradient(head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template)
        return gain



    # here we use the  width-first search to add body predicates
    # we assume that the number of important rules should be smaller thant K
    # we assume that the length of the body predicates should be smaller thant L
    def get_feature_sum_for_screen(self, dataset, head_predicate_idx, template):
        sample_ID_batch = list(dataset.keys())[:self.batch_size]
        feature_sum = 0
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
            effect_formula = []
            feature_formula= []

            for cur_time in data_sample[head_predicate_idx]['time'][1:]:
                feature_formula.append(self.get_feature(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                        history=data_sample, template=template))
                effect_formula.append(self.get_formula_effect(cur_time=cur_time, head_predicate_idx=head_predicate_idx,
                                                        history=data_sample, template=template))
            if len(feature_formula) != 0:
                feature_sum += torch.sum(torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0))
                print("data_sample=", data_sample)
                print("effect_formula =", torch.cat(effect_formula, dim=0))
                print("feature_formula=", torch.cat(feature_formula, dim=0))
        return feature_sum
                
            
    def initialize_rule_set(self, head_predicate_idx, dataset, T_max):
        new_rule_table = {}
        new_rule_table[head_predicate_idx] = {}
        new_rule_table[head_predicate_idx]['body_predicate_idx'] = []
        new_rule_table[head_predicate_idx]['body_predicate_sign'] = []  # use 1 to indicate True; use 0 to indicate False
        new_rule_table[head_predicate_idx]['head_predicate_sign'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_idx'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_type'] = []
        new_rule_table[head_predicate_idx]['performance_gain'] = []
        new_rule_table[head_predicate_idx]['weight'] = []

        ## search for the new rule from by minimizing the gradient of the log-likelihood
        
        for head_predicate_sign in [1, 0]:  # consider head_predicate_sign = 1/0
            for body_predicate_sign in [1, 0]:
                for body_predicate_idx in self.predicate_set:  
                    if body_predicate_idx == head_predicate_idx: # all the other predicates, excluding the head predicate, can be the potential body predicates
                        continue
                    for temporal_relation_type in [self.BEFORE, self.EQUAL, self.AFTER]:
                        # create new rule
                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append([body_predicate_idx])
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append([body_predicate_sign])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append([head_predicate_sign])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append([(body_predicate_idx, head_predicate_idx)])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append([temporal_relation_type])
                        new_rule_table[head_predicate_idx]['weight'].append(torch.autograd.Variable((torch.ones(1) ).double(), requires_grad=True))
                        
                        #temporally add new rule, to get likelihood.
                        
                        self.logic_template[head_predicate_idx][self.num_formula] = {}
                        for k in new_rule_table[head_predicate_idx].keys():
                            if k != 'weight' and k !='performance_gain':
                                #print(k, new_rule_table[head_predicate_idx][k])
                                self.logic_template[head_predicate_idx][self.num_formula][k] = new_rule_table[head_predicate_idx][k][-1]
                        self.model_parameter[head_predicate_idx][self.num_formula] = {}
                        self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = new_rule_table[head_predicate_idx]['weight'][-1]
                        self.num_formula +=1

                        feature_sum = self.get_feature_sum_for_screen(dataset, head_predicate_idx, template=self.logic_template[head_predicate_idx][self.num_formula-1])
                        #record the log-likelihood_gradient in performance gain
                        #gain  = self.optimize_log_likelihood(dataset, T_max)
                        
                        print("Current rule is:", flush=1)
                        self.print_rule()
                        print("feature sum is", feature_sum, flush=1)
                        print("-------------",flush=1)

                        new_rule_table[head_predicate_idx]['performance_gain'].append(feature_sum)

                        #remove the new rule.
                        self.num_formula -=1
                        self.logic_template[head_predicate_idx][self.num_formula] = {}
                        self.model_parameter[head_predicate_idx][self.num_formula] = {}

        idx = np.argmax(new_rule_table[head_predicate_idx]['performance_gain'])


        # add new rule
        self.logic_template[head_predicate_idx][self.num_formula] = {}
        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]
        # add model parameter
        self.model_parameter[head_predicate_idx][self.num_formula] = {}
        self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = new_rule_table[head_predicate_idx]['weight'][idx]
        self.num_formula += 1

        #update params
        self.optimize_log_likelihood(dataset, T_max)

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
        print("start generate simple rule")
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

        print("start searching.", flush=1)
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

                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append(new_rule_template[head_predicate_idx]['body_predicate_idx'] )
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append(new_rule_template[head_predicate_idx]['body_predicate_sign'])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append(new_rule_template[head_predicate_idx]['head_predicate_sign'])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(new_rule_template[head_predicate_idx]['temporal_relation_idx'])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(new_rule_template[head_predicate_idx]['temporal_relation_type'])
                        #record the log-likelihood_gradient in performance gain
                        gain  = self.optimize_log_likelihood_gradient(head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template[head_predicate_idx])
                        new_rule_table[head_predicate_idx]['performance_gain'].append(gain)

        if len(new_rule_table[head_predicate_idx]['performance_gain']) == 0: # No candidate rule generated.
            return False

        # choose the logic rule that leads to the optimal log-likelihood
        idx = np.argmax(new_rule_table[head_predicate_idx]['performance_gain'])
        best_gain = new_rule_table[head_predicate_idx]['performance_gain'][idx]
        threshold = 0.1
        if  best_gain > threshold:
            # add new rule
            self.logic_template[head_predicate_idx][self.num_formula] = {}
            self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
            self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
            self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
            self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
            self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

            # add model parameter
            self.model_parameter[head_predicate_idx][self.num_formula] = {}
            self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)
            # update model parameter
            self.optimize_log_likelihood(dataset, T_max)
            self.num_formula += 1
            return True
        else:
            return False


    def add_one_predicate_to_existing_rule(self, head_predicate_idx, dataset, T_max, existing_rule_template):
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
                        
                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append(new_rule_template[head_predicate_idx]['body_predicate_idx'] )
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append(new_rule_template[head_predicate_idx]['body_predicate_sign'])
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append(new_rule_template[head_predicate_idx]['head_predicate_sign'])
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(new_rule_template[head_predicate_idx]['temporal_relation_idx'])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(new_rule_template[head_predicate_idx]['temporal_relation_type'])

                        #calculate gradient, store as gain.
                        gain  = self.optimize_log_likelihood_gradient(head_predicate_idx, dataset, T_max, intensity_log_gradient, intensity_integral_gradient_grid, new_rule_template[head_predicate_idx])
                        new_rule_table[head_predicate_idx]['performance_gain'].append(gain)

        if len(new_rule_table[head_predicate_idx]['performance_gain']) == 0: # No candidate rule generated.
            return False

        # choose the logic rule that leads to the optimal log-likelihood
        idx = np.argmax(new_rule_table[head_predicate_idx]['performance_gain'])
        best_gain = new_rule_table[head_predicate_idx]['performance_gain'][idx]
        threshold = 0.1
        if  best_gain > threshold:
            # add new rule
            self.logic_template[head_predicate_idx][self.num_formula] = {}
            self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
            self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
            self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
            self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
            self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

            # add model parameter
            self.model_parameter[head_predicate_idx][self.num_formula] = {}
            self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)
            # update model parameter
            self.optimize_log_likelihood(dataset, T_max)
            self.num_formula += 1
            return True
        else:
            return False


    def prune_rules_with_small_weights(self):
        # if num_formula -=1, then hwo to move existing formulas?
        # maybe only prune when learning finishes.
        # TODO
        pass


    def search_algorithm(self, head_predicate_idx, dataset, T_max):
        self.initialize_rule_set(head_predicate_idx, dataset, T_max)
        print("Initialize with this rule:")
        self.print_rule()
        #Begin Breadth(width) First Search
        #generate new rule from scratch
        while self.num_formula < self.max_num_rule:
            rule_accepted = self.generate_rule_via_column_generation(head_predicate_idx, dataset, T_max)
            if not rule_accepted:
                break
            else:
                print("Add a simple rule. Current rule set is:")
                self.print_rule()
        
        #generate new rule by extending existing rules
        for cur_body_length in range(1, self.max_rule_body_length + 1):
            for existing_rule_template in list(self.logic_template[head_predicate_idx].values()):
                if self.num_formula >= self.max_num_rule:
                    break
                if len(existing_rule_template['body_predicate_idx']) == cur_body_length: 
                    rule_accepted = self.add_one_predicate_to_existing_rule(head_predicate_idx, dataset, T_max, existing_rule_template)
                    if rule_accepted:
                        print("Extend an existing rule. Current rule set is:")
                        self.print_rule()

        print("Train finished, Final rule set is:")
        self.print_rule()
        #self.prune_rule_by_small_weights()

    def print_rule(self):
        for head_predicate_idx, rules in self.logic_template.items():
            print("Head = {}, base = {:.4f}".format(self.predicate_notation[head_predicate_idx], self.model_parameter[head_predicate_idx]['base'].data[0]))
            for rule_id, rule in rules.items():
                body_predicate_idx = rule['body_predicate_idx']
                body_predicate_sign = rule['body_predicate_sign']
                head_predicate_sign = rule['head_predicate_sign'][0]
                temporal_relation_idx = rule['temporal_relation_idx']
                temporal_relation_type = rule['temporal_relation_type']
                rule_str = "Rule{}: ".format(rule_id)
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
                weight = self.model_parameter[head_predicate_idx][rule_id]['weight'].data[0]
                rule_str += ", weight={:.4f}".format(weight)
                print(rule_str)

                
                




if __name__ == "__main__":
    head_predicate_idx = [3,4]
    model = Logic_Learning_Model(head_predicate_idx = head_predicate_idx)
    num_samples = 5000
    T_max = 10
    dataset = np.load('data.npy', allow_pickle='TRUE').item()

    small_dataset = {i:dataset[i] for i in range(500)}
    model.search_algorithm(head_predicate_idx[0], small_dataset, T_max)
    #model.model_parameter = {4: {'base': torch.tensor([-0.2000], dtype=torch.float64, requires_grad=True), 0: {'weight': torch.tensor([0.0100], dtype=torch.float64, requires_grad=True)}, 1: {'weight': torch.tensor([0.0100], dtype=torch.float64, requires_grad=True)}, 2: {'weight': torch.tensor([0.0100], dtype=torch.float64, requires_grad=True)}, 3: {'weight': torch.tensor([0.0100], dtype=torch.float64, requires_grad=True)}, 4: {'weight': torch.tensor([0.0100], dtype=torch.float64, requires_grad=True)}}}
    #model.logic_template = {4: {0: {'body_predicate_idx': [0], 'body_predicate_sign': [-1], 'head_predicate_sign': [-1], 'temporal_relation_idx': [(0, 4)], 'temporal_relation_type': ['EQUAL']}, 1: {'body_predicate_idx': [2], 'body_predicate_sign': [1], 'head_predicate_sign': [1], 'temporal_relation_idx': [(2, 4)], 'temporal_relation_type': ['BEFORE']}, 2: {'body_predicate_idx': [3], 'body_predicate_sign': [1], 'head_predicate_sign': [1], 'temporal_relation_idx': [(3, 4)], 'temporal_relation_type': ['EQUAL']}, 3: {'body_predicate_idx': [2], 'body_predicate_sign': [1], 'head_predicate_sign': [-1], 'temporal_relation_idx': [(2, 4)], 'temporal_relation_type': ['BEFORE']}, 4: {'body_predicate_idx': [3], 'body_predicate_sign': [1], 'head_predicate_sign': [-1], 'temporal_relation_idx': [(3, 4)], 'temporal_relation_type': ['EQUAL']}}}
    #print(model.logic_template)
    #print(model.model_parameter)
    model.print_rule()
    with open("model.pkl",'wb') as f:
        pickle.dump(model, f)







