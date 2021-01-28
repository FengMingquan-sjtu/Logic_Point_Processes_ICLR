import numpy as np
import itertools
import random
from generate_synthetic_data import Logic_Model_Generator
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
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
        self.head_predicate_set = head_predicate_idx  # the index set of only one head predicates

        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = 0.5
        self.integral_resolution = 0.03
        self.decay_rate = 1
        self.max_rule_body_length = 3 #
        self.max_num_rule = 3
        
        self.batch_size = 20
        self.num_iter  = 1
        self.master_lr = 0.03
        self.sub_lr = 0.03
        
        #claim parameters 
        self.model_parameter = {}

        for idx in head_predicate_idx:
            self.model_parameter[idx] = {}
            self.model_parameter[idx]['base'] = torch.autograd.Variable((torch.ones(1) * -0.2).double(), requires_grad=True)




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
        intensity = torch.exp(torch.cat(weight_formula, dim=0))/torch.sum(torch.exp(torch.cat(weight_formula, dim=0)), dim=0) * torch.cat(feature_formula, dim=0) * torch.cat(effect_formula, dim=0)
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


    def optimize_log_likelihood(self, dataset, T_max, new_rule_parameter):
        optimizer = optim.Adam(new_rule_parameter, lr=self.master_lr)
        log_likelihood = 0
        for i in range(self.num_iter):
            sample_ID_list = list(dataset.keys())
            random.shuffle(sample_ID_list)
            for batch_idx in range(len(sample_ID_list)//self.batch_size):
                sample_ID_batch = sample_ID_list[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                loss = -self.log_likelihood(dataset, sample_ID_batch, T_max)
                log_likelihood = - loss.data[0]
                loss.backward()
                optimizer.step()

        #use the last batch log_likelihood
        return log_likelihood



    def intensity_log_gradient(self, head_predicate_idx, data_sample):
        intensity_transition = []
        for t in data_sample[head_predicate_idx]['time'][1:]:
            cur_intensity = self.intensity(t, head_predicate_idx, data_sample)
            intensity_transition.append(cur_intensity)
        if len(intensity_transition) == 0:  # only survival term, not event happens
            log_gradient = torch.tensor([0], dtype=torch.float64)
        else:
            log_gradient = torch.cat(intensity_transition, dim=0).pow(-1)
        return log_gradient

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


    def log_likelihood_gradient(self, head_predicate_idx, dataset, sample_ID_batch, T_max, new_rule_template, log_intensity_gradient, integral_gradient_grid):
        log_likelihood_gradient = torch.tensor([0], dtype=torch.float64)
        # iterate over samples
        for sample_ID in sample_ID_batch:
            # iterate over head predicates; each predicate corresponds to one intensity
            data_sample = dataset[sample_ID]
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
            log_likelihood_gradient += torch.sum(log_intensity_gradient * torch.cat(new_feature_transition_times, dim=0)) - \
                                       torch.sum(integral_gradient_grid * torch.cat(new_feature_grid_times, dim=0), dim=0)
        return log_likelihood_gradient


    def optimize_log_likelihood_gradient(self, head_predicate_idx, dataset, T_max, new_rule_template, log_intensity_gradient, integral_gradient_grid, new_rule_parameter):
        optimizer = optim.Adam(new_rule_parameter, lr=self.sub_lr)
        gain = 0
        for i in range(self.num_iter):
            sample_ID_list = list(dataset.keys())
            random.shuffle(sample_ID_list)
            for batch_idx in range(len(sample_ID_list)//self.batch_size):
                sample_ID_batch = sample_ID_list[batch_idx*self.batch_size : (batch_idx+1)*self.batch_size]
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                loss = -self.log_likelihood_gradient(head_predicate_idx, dataset, sample_ID_batch, T_max, new_rule_template, log_intensity_gradient, integral_gradient_grid)
                gain = -loss.data[0]
                loss.backward()
                optimizer.step()

        #use the last batch score
        return gain

    def get_feature_unknown_temporal_relation(self, cur_time, head_predicate_idx, history, template):
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

    # here we use the  width-first search to add body predicates
    # we assume that the number of important rules should be smaller thant K
    # we assume that the length of the body predicates should be smaller thant L

    def initialize_rule_set(self, head_predicate_idx, dataset, T_max):
        ## start from an empty rule set
        new_rule_table = {}
        new_rule_table[head_predicate_idx] = {}
        new_rule_table[head_predicate_idx]['body_predicate_idx'] = []
        new_rule_table[head_predicate_idx]['body_predicate_sign'] = []  # use 1 to indicate True; use -1 to indicate False
        new_rule_table[head_predicate_idx]['head_predicate_sign'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_idx'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_type'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_parameter']=[]
        new_rule_table[head_predicate_idx]['performance_gain'] = []
        new_rule_table[head_predicate_idx]['weight'] = []
        new_rule_table[head_predicate_idx]['base'] = []

        ## search for the first rule from scratch by maximizing the log-likelihood
        ## provide an initial screening of the candidate variable

        self.num_formula = 0

        for head_predicate_sign in [1, -1]:  # consider head_predicate_sign = 1/-1
            for body_predicate_sign in [1, -1]:
                for body_predicate_idx in self.predicate_set.remove(head_predicate_idx): # all the other predicates, excluding the head predicate, can be the potential body predicates
                        new_rule_table[head_predicate_idx]['body_predicate_idx'].append(body_predicate_idx)
                        new_rule_table[head_predicate_idx]['body_predicate_sign'].append(body_predicate_sign)
                        new_rule_table[head_predicate_idx]['head_predicate_sign'].append(head_predicate_sign)

                        # claim variables here
                        new_rule_parameter = {}
                        new_rule_parameter['base'] = torch.autograd.Variable((torch.ones(1) * -0.0).double(), requires_grad=True)
                        new_rule_parameter['weight'] = torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)
                        new_rule_parameter['temporal_relation_weight'] = torch.autograd.Variable((torch.ones(4)/4).double(), requires_grad=True) #delete softmax, since optimizer can not optimize non-leaf tensor

                        ## maximize log-likelihood, learn model parameters, and record the log-likelihood in performance gain


                        log_likelihood = self.optimize_log_likelihood(dataset, T_max, new_rule_parameter) # TODO[Done]: revise this function, note that output is a list. 
                        weight = new_rule_parameter['weight'].data[0]
                        base = new_rule_parameter['base'].data[0]
                        temporal_relation_parameter = new_rule_parameter['temporal_relation_weight'].data.numpy()

                        new_rule_table[head_predicate_idx]['performance_gain'].append(log_likelihood)
                        new_rule_table[head_predicate_idx]['weight'].append(weight)
                        new_rule_table[head_predicate_idx]['base'].append(base)
                        new_rule_table[head_predicate_idx]['temporal_relation_parameter'].append(temporal_relation_parameter)

                        if np.argmax(temporal_relation_parameter) == 0: # Before
                            new_rule_table[head_predicate_idx]['temporal_relation_idx'].append([body_predicate_idx, head_predicate_idx])
                            new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.BEFORE)
                        if np.argmax(temporal_relation_parameter) == 1: # EQUAL
                            new_rule_table[head_predicate_idx]['temporal_relation_idx'].append([body_predicate_idx, head_predicate_idx])
                            new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.EQUAL)
                        if np.argmax(temporal_relation_parameter) == 2: # AFTER
                            new_rule_table[head_predicate_idx]['temporal_relation_idx'].append([body_predicate_idx, head_predicate_idx])
                            new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.AFTER)
                        if np.argmax(temporal_relation_parameter) == 3:  # None
                            new_rule_table[head_predicate_idx]['temporal_relation_idx'].append([])
                            new_rule_table[head_predicate_idx]['temporal_relation_type'].append(None)

        ### choose the logic rule that leads to the optimal log-likelihood

        idx = np.argmax([head_predicate_idx]['performance_gain'])

        # add new rule
        self.logic_template[head_predicate_idx][self.num_formula] = {}
        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_idx'] = new_rule_table[head_predicate_idx]['body_predicate_idx'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['body_predicate_sign'] = new_rule_table[head_predicate_idx]['body_predicate_sign'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['head_predicate_sign'] = new_rule_table[head_predicate_idx]['head_predicate_sign'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_idx'] = new_rule_table[head_predicate_idx]['temporal_relation_idx'][idx]
        self.logic_template[head_predicate_idx][self.num_formula]['temporal_relation_type'] = new_rule_table[head_predicate_idx]['temporal_relation_type'][idx]

        # add model parameter
        self.model_parameter = {}
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.autograd.Variable( torch.Tensor([new_rule_table[head_predicate_idx]['base'][idx]], dtype=torch.float64), requires_grad=True)
        self.model_parameter[head_predicate_idx][self.num_formula] = {}
        self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable(torch.Tensor([new_rule_table[head_predicate_idx]['weight'][idx]], dtype=torch.float64),requires_grad=True)

        self.num_formula += 1
        return


    def generate_rule_via_column_generation(self, head_predicate_idx, dataset, T_max):
        ## generate one new rule to the rule set by columns generation
        new_rule_table = {}
        new_rule_table[head_predicate_idx] = {}
        new_rule_table[head_predicate_idx]['body_predicate_idx'] = []
        new_rule_table[head_predicate_idx][
            'body_predicate_sign'] = []  # use 1 to indicate True; use -1 to indicate False
        new_rule_table[head_predicate_idx]['head_predicate_sign'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_idx'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_type'] = []
        new_rule_table[head_predicate_idx]['temporal_relation_parameter']=[]
        new_rule_table[head_predicate_idx]['performance_gain'] = []


        ## search for the new rule from by minimizing the gradient of the log-likelihood

        for head_predicate_sign in [1, -1]:  # consider head_predicate_sign = 1/-1
            for body_predicate_sign in [1, -1]:
                for body_predicate_idx in self.predicate_set.remove(head_predicate_idx):  # all the other predicates, excluding the head predicate, can be the potential body predicates
                    new_rule_table[head_predicate_idx]['body_predicate_idx'].append(body_predicate_idx)
                    new_rule_table[head_predicate_idx]['body_predicate_sign'].append(body_predicate_sign)
                    new_rule_table[head_predicate_idx]['head_predicate_sign'].append(head_predicate_sign)

                    # maximize log-likelihood, learn model parameters, and record the log-likelihood in performance gain
                    # compute the log_intensity_gradient, integral_gradient_grid using existing rules
                    intensity_log_gradient = self.intensity_log_gradient(head_predicate_idx, data_sample)
                    intensity_integral_gradient_grid = self.intensity_integral_gradient(head_predicate_idx, data_sample, T_max)

                    # claim variables here
                    new_rule_parameter = {}
                    new_rule_parameter['weight'] = torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)
                    new_rule_parameter['temporal_relation_weight'] = torch.autograd.Variable((torch.ones(4)/4).double(), requires_grad=True)

                    new_rule_template = {}
                    new_rule_template[head_predicate_idx]= {}
                    new_rule_template[head_predicate_idx]['body_predicate_idx'] = [body_predicate_idx]
                    new_rule_template[head_predicate_idx]['body_predicate_sign'] = [body_predicate_sign]  # use 1 to indicate True; use -1 to indicate False
                    new_rule_template[head_predicate_idx]['head_predicate_sign'] = [head_predicate_sign]


                    gain  = self.optimize_log_likelihood_gradient(head_predicate_idx,dataset, sample_ID_batch,T_max, new_rule_template,
                                                                                              intensity_log_gradient,
                                                                                              intensity_integral_gradient_grid,
                                                                                              new_rule_parameter)
                    temporal_relation_parameter = new_rule_parameter['temporal_relation_weight'].data.numpy()

                    new_rule_table[head_predicate_idx]['performance_gain'].append(gain)
                    new_rule_table[head_predicate_idx]['temporal_relation_parameter'].append(temporal_relation_parameter)

                    if np.argmax(temporal_relation_parameter) == 0:  # Before
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(
                            [body_predicate_idx, head_predicate_idx])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.BEFORE)
                    if np.argmax(temporal_relation_parameter) == 1:  # EQUAL
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(
                            [body_predicate_idx, head_predicate_idx])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.EQUAL)
                    if np.argmax(temporal_relation_parameter) == 2:  # AFTER
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(
                            [body_predicate_idx, head_predicate_idx])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.AFTER)
                    if np.argmax(temporal_relation_parameter) == 3:  # None
                        new_rule_table[head_predicate_idx]['temporal_relation_idx'].append([])
                        new_rule_table[head_predicate_idx]['temporal_relation_type'].append(None)

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
            self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable(torch.Tensor([new_rule_table[head_predicate_idx]['weight'][idx]], dtype=torch.float64), requires_grad=True)
            # update model parameter
            self.optimize_log_likelihood(dataset, sample_ID_batch, T_max, optimizer)
            self.num_formula += 1
            return True
        else:
            return False


    def add_one_predicate_to_existing_rule(self, head_predicate_idx, dataset, sample_ID_batch, T_max, existing_rule_template):
        ## increase the length of the existing rules
        ## generate one new rule to the rule set by columns generation
        new_rule_table = {}
        new_rule_table[head_predicate_idx] = {}
        new_rule_table[head_predicate_idx]['body_predicate_idx'] = existing_rule_template[head_predicate_idx]['body_predicate_idx']
        new_rule_table[head_predicate_idx]['body_predicate_sign'] = existing_rule_template[head_predicate_idx]['body_predicate_sign']
        new_rule_table[head_predicate_idx]['temporal_relation_idx'] = existing_rule_template[head_predicate_idx]['temporal_relation_idx']
        new_rule_table[head_predicate_idx]['temporal_relation_type'] = existing_rule_template[head_predicate_idx]['temporal_relation_type']
        new_rule_table[head_predicate_idx]['temporal_relation_parameter'] = []
        new_rule_table[head_predicate_idx]['performance_gain'] = []

        ## search for the new rule from by minimizing the gradient of the log-likelihood

        for body_predicate_sign in [1, -1]:
            for body_predicate_idx in self.predicate_set.remove([existing_rule_template['head_predicate_idx'], existing_rule_template['body_predicate_idx']]):
                new_rule_table[head_predicate_idx]['body_predicate_idx'].append(body_predicate_idx)
                new_rule_table[head_predicate_idx]['body_predicate_sign'].append(body_predicate_sign)
                # maximize log-likelihood, learn model parameters, and record the log-likelihood in performance gain
                # compute the log_intensity_gradient, integral_gradient_grid using existing rules
                intensity_log_gradient = self.intensity_log_gradient(head_predicate_idx, data_sample)
                intensity_integral_gradient_grid = self.intensity_integral_gradient(head_predicate_idx, data_sample,T_max)

                # claim variables here
                new_rule_parameter = {}
                new_rule_parameter['weight'] = torch.autograd.Variable((torch.ones(1) * 0.01).double(), requires_grad=True)
                num_existing_predicate = len(existing_rule_template['head_predicate_idx']) + len(existing_rule_template['body_predicate_idx'])
                new_rule_parameter['temporal_relation_weight'] = torch.autograd.Variable((torch.ones(num_existing_predicate, 4)/4).double(), requires_grad=True)

                new_rule_template = existing_rule_template
                new_rule_template[head_predicate_idx]['body_predicate_idx'].append(body_predicate_idx)
                new_rule_template[head_predicate_idx]['body_predicate_sign'].append(body_predicate_sign) # use 1 to indicate True; use -1 to indicate False

                gain, temporal_relation_parameter = self.optimize_log_likelihood_gradient(head_predicate_idx,
                                                                                          dataset, sample_ID_batch,
                                                                                          T_max, new_rule_template,
                                                                                          intensity_log_gradient,
                                                                                          intensity_integral_gradient_grid,
                                                                                          new_rule_parameter)
                temporal_relation_parameter = new_rule_parameter['temporal_relation_weight'].data.numpy()

                new_rule_table[head_predicate_idx]['performance_gain'].append(gain)
                new_rule_table[head_predicate_idx]['temporal_relation_parameter'].append(
                    temporal_relation_parameter)

                if np.argmax(temporal_relation_parameter) == 0:  # Before
                    new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(
                        [body_predicate_idx, head_predicate_idx])
                    new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.BEFORE)
                if np.argmax(temporal_relation_parameter) == 1:  # EQUAL
                    new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(
                        [body_predicate_idx, head_predicate_idx])
                    new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.EQUAL)
                if np.argmax(temporal_relation_parameter) == 2:  # AFTER
                    new_rule_table[head_predicate_idx]['temporal_relation_idx'].append(
                        [body_predicate_idx, head_predicate_idx])
                    new_rule_table[head_predicate_idx]['temporal_relation_type'].append(self.AFTER)
                if np.argmax(temporal_relation_parameter) == 3:  # None
                    new_rule_table[head_predicate_idx]['temporal_relation_idx'].append([])
                    new_rule_table[head_predicate_idx]['temporal_relation_type'].append(None)

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
            self.model_parameter[head_predicate_idx][self.num_formula]['weight'] = torch.autograd.Variable(torch.Tensor([new_rule_table[head_predicate_idx]['weight'][idx]], dtype=torch.float64), requires_grad=True)
            # update model parameter
            self.optimize_log_likelihood(dataset, sample_ID_batch, T_max, optimizer)
            self.num_formula += 1
            return True
        else:
            return False


    def prune_rules_with_small_weights(self):
        # TODO
        pass



    def search_algorithm(self, head_predicate_idx, dataset, sample_ID_batch, T_max):

        self.initialize_rule_set(head_predicate_idx, dataset, sample_ID_batch, T_max)

        #Begin Breadth(width) First Search

        #generate new rule from scratch
        while self.num_formula <= self.max_num_rule:
            rule_accepted = self.generate_rule_via_column_generation(head_predicate_idx, dataset,sample_ID_batch, T_max)
            if not rule_accepted:
                break
        
        #generate new rule by extending existing rules
        cur_rule_boldy_length = 1
        last_level_formula_idx = 0
        while cur_rule_boldy_length < self.max_rule_body_length:
            cur_rule_boldy_length +=1
            cur_level_formula_idx = self.num_formula - 1
            for formula_idx in range(last_level_formula_idx, cur_level_formula_idx + 1):
                existing_rule_template = self.logic_template[head_predicate_idx][formula_idx]
                self.add_one_predicate_to_existing_rule(head_predicate_idx, dataset, sample_ID_batch, T_max, existing_rule_template)
            last_level_formula_idx = cur_level_formula_idx

        self.prune_rule_by_small_weights()





if __name__ == "__main__":
    logic_model_generator = Logic_Model_Generator()
    num_samples = 5000
    time_horizon = 10
    batch_size = 20
    num_batch = num_samples / batch_size
    data = np.load('data.npy', allow_pickle='TRUE').item()

    model = Logic_Learning_Model()

    
    for iter in range(10000):
        for batch_idx in np.arange(0, num_batch, 1):
            indices = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
            loss = logic_model.optimize_log_likelihood(data, indices, time_horizon, optimizer)
            print('loss is', loss)
            print('base 1 of logic 1 is', logic_model.model_parameter[3]['base'])
            print('weight 1 of logic 1 is', torch.exp(logic_model.model_parameter[3][0]['weight'])/(torch.exp(logic_model.model_parameter[3][0]['weight']) +torch.exp(logic_model.model_parameter[3][1]['weight'])))
            print('weight 2 of logic 1 is', torch.exp(logic_model.model_parameter[3][1]['weight'])/(torch.exp(logic_model.model_parameter[3][0]['weight']) +torch.exp(logic_model.model_parameter[3][1]['weight'])))
            print('base 2 is', logic_model.model_parameter[4]['base'])
            print('weight 3 is', torch.exp(logic_model.model_parameter[4][0]['weight'])/torch.exp(logic_model.model_parameter[4][0]['weight']))






