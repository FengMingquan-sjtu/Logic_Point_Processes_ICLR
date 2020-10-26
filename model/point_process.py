from collections import OrderedDict
import itertools
from typing import List,Tuple,Dict,Any

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from logic import Logic

class Point_Process:
    """functions used in PP.
    """
    def __init__(self, args):
        self.logic = Logic(args)
        self.args = args
        self.template = {t:self.logic.get_template(t) for t in args.target_predicate}
        # initialize parameters
        self.num_formula = self.logic.logic.num_formula
        self.num_predicate = self.logic.logic.num_predicate
        self._parameters = OrderedDict()
        self._parameters["weight"] = torch.autograd.Variable((torch.rand(self.num_formula) * self.args.init_weight_range).double(), requires_grad=True)
        #self._parameters["weight"] = torch.autograd.Variable((torch.rand(num_formula)).double(), requires_grad=True)
        self._parameters["base"] = torch.autograd.Variable((torch.rand(self.num_predicate)* self.args.init_weight_range).double(), requires_grad=True)
        #self._parameters["weight"] = torch.autograd.Variable((torch.ones(num_formula)* 0.2).double(), requires_grad=True)
        #self._parameters["base"] = torch.autograd.Variable((torch.ones(num_predicate)* 0.2).double(), requires_grad=True)
        # cache
        self.feature_cache = dict()
    
    def set_parameters(self, w:float, b:float, requires_grad:bool=True):
        self._parameters["weight"] = torch.autograd.Variable((torch.ones(self.num_formula)* w).double(), requires_grad=requires_grad)
        self._parameters["base"] = torch.autograd.Variable((torch.ones(self.num_predicate)* b).double(), requires_grad=requires_grad)
        
    def _check_state(self, seq:Dict[str, List], cur_time:float) -> int:
        """check state of seq at cur_time
        """
        if seq['time'] ==[]:
            return 0
        else:
            ind = np.sum(cur_time >= np.asarray(seq['time'])) - 1
            cur_state = seq['state'][ind]
            return cur_state
    
    def _get_time_window(self, formula_ind:int) -> float:
        if self.args.dataset_name == "mimic":
            if formula_ind in [43, 48]:
                time_window = self.args.time_window_sym
            else:
                time_window = self.args.time_window_drug
        elif self.args.dataset_name == "synthetic":
            time_window = self.args.synthetic_time_window
        else:
            raise ValueError("time window is undefined for dataset_name = '{}'".format(self.args.dataset_name))
        return time_window

    def _get_filtered_transition_time(self, data:Dict, time_window:float, t:float, neighbor_ind:List, neighbor_combination:np.ndarray) -> Tuple[List, bool]: 
        transition_time_list = list()
        is_early_stop = False
        for idx,neighbor_ind_ in enumerate(neighbor_ind):
            transition_time = np.array(data[neighbor_ind_]['time'])
            transition_state = np.array(data[neighbor_ind_]['state'])
            # only use history that falls in range [t - time_window, t)
            # and neighbors should follow neighbor_combination
            mask = (transition_time >= t - time_window) * (transition_time <= t) * (transition_state == neighbor_combination[idx])
            transition_time = transition_time[mask]
            transition_time_list.append(transition_time)
            if len(transition_time) == 0:
                is_early_stop = True
                break
        return transition_time_list, is_early_stop
    
    def _get_history_cnt(self, target_ind_in_predicate:int, time_template:np.ndarray, transition_time_list:List, t:float) -> float:
        transition_time_list_ = transition_time_list.copy() #copy, avoid modifing input
        transition_time_list_.insert(target_ind_in_predicate, np.array([t,]))
        # NOTE: we consider simple case that only A->Others have time relation.
        # time_template denotes relations of A->Others.
        # transition_time_list_[0] is A's time
        t_a = np.expand_dims(transition_time_list_[0], axis=0)
        cnt_array = np.ones(len(transition_time_list_[0])) #cnt.shape = t_a.shape[1] = len(A's time list)
        for idx,transition_time in enumerate(transition_time_list_):
            if idx == 0: # A has no relation with itself. 
                continue
            else:
                t_ = np.expand_dims(transition_time_list_[idx], axis=1)
                if time_template[idx] == self.logic.logic.BEFORE:
                    mask = (t_ - t_a) >= self.args.time_tolerence
                elif time_template[idx] == self.logic.logic.EQUAL:
                    mask = abs(t_ - t_a) < self.args.time_tolerence
                else:
                    raise ValueError("Unrecognized relation '{}'".format(relation[idx]))
                if idx == target_ind_in_predicate:
                    mask = mask *  np.exp( - abs(t_ - t_a) * self.args.time_decay_rate ) # time decay kernel.
                cnt = np.sum(mask, axis=0) #cnt.shape = t_a.shape[1]
                cnt_array *= cnt
        history_cnt = np.sum(cnt_array)
        return history_cnt

    def get_feature(self, t:float, dataset:Dict, sample_ID:int, target_predicate:int) -> torch.Tensor:
        if (target_predicate,sample_ID,t) in self.feature_cache:
            #Notice that sample_ID in testing and training should be different
            feature_list = self.feature_cache[(target_predicate,sample_ID,t)]
        else:
            #### begin calculating feature ####
            formula_ind_list = list(self.template[target_predicate].keys()) # extract formulas related to target_predicate
            cur_state = self._check_state(dataset[sample_ID][target_predicate], t)  # cur_state is either 0 or 1
            feature_list = list()
            # collect evidence
            for formula_ind in formula_ind_list:
                # read info from template
                template = self.template[target_predicate][formula_ind]
                neighbor_ind, neighbor_combination, target_ind_in_predicate, time_template = template['neighbor_ind'],template['neighbor_combination'], template['target_ind_in_predicate'], template['time_template']
                formula_effect = template['formula_effect'][cur_state]

                time_window = self._get_time_window(formula_ind=formula_ind)
                transition_time_list, is_early_stop = self._get_filtered_transition_time(data=dataset[sample_ID], time_window=time_window, t=t, neighbor_ind=neighbor_ind, neighbor_combination=neighbor_combination)
                
                if is_early_stop:
                    history_cnt = 0
                else:
                    history_cnt = self._get_history_cnt(target_ind_in_predicate, time_template, transition_time_list, t)           
                feature = torch.tensor(history_cnt * formula_effect).double()
                feature_list.append(feature)
            feature_list = torch.tensor(feature_list)
            #print(feature_list)
            self.feature_cache[(target_predicate,sample_ID,t)] = feature_list
            #### end calculating feature ####
        return feature_list

    def intensity(self, t:float, dataset:Dict, sample_ID:int, target_predicate:int) -> torch.Tensor:
        """Calculate intensity of target_predicate given dataset[sample_ID], following Eq(9).
        """
        feature_list = self.get_feature(t, dataset, sample_ID, target_predicate)
        formula_ind_list = list(self.template[target_predicate].keys()) # extract formulas related to target_predicate
        #weight = F.softmax(self._parameters["weight"][formula_ind_list], dim=0) 
        weight = self._parameters["weight"][formula_ind_list]
        #weight = torch.clamp(self._parameters["weight"][formula_ind_list], min=0.001, max=0.99)
        base = self._parameters["base"][target_predicate]
        f = torch.add(torch.sum(torch.mul(feature_list, weight)), base)
        intensity = self._non_negative_map(f)
        intensity = intensity.reshape((1,)) # convert scalar to tensor, for conveninent grad
        return intensity

    def _non_negative_map(self, f:torch.Tensor) -> torch.Tensor:
        if self.args.non_negative_map == "exp":
            intensity = torch.exp(f)
        elif self.args.non_negative_map == "max":
            intensity = torch.clamp(f, min=0.0001)
        else:
            raise ValueError("Undefiend non_negative_map name {}".format(self.args.non_negative_map))
        return intensity
    
    def intensity_log_sum(self):
        pass
    def intensity_integral(self):
        pass
    def log_likelihood(self):
        pass
    def predict_survival(self):
        pass