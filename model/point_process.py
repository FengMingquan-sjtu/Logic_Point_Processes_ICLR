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
        self._parameters["base"] = torch.autograd.Variable((torch.rand(self.num_predicate)* self.args.init_weight_range).double(), requires_grad=True)
        self.feature_cache = dict()
        self.feature_integral_cache = dict()
    
    def set_parameters(self, w:float, b:float, requires_grad:bool=True):
        self._parameters["weight"] = torch.autograd.Variable((torch.ones(self.num_formula)* w).double(), requires_grad=requires_grad)
        self._parameters["base"] = torch.autograd.Variable((torch.ones(self.num_predicate)* b).double(), requires_grad=requires_grad)
        
    def _check_state(self, seq:Dict[str, List], cur_time:float) -> int:
        """check state of seq at cur_time
        """
        if len(seq['time']) == 0:
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
        elif self.args.dataset_name in ["synthetic","toy"]:
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
            #Notice that sample_ID in testing and training should be different, for easy cache.
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
    
    def intensity_log_sum(self, dataset, sample_ID, target_predicate):
        intensity_list = list()
        is_duration_pred =  self.logic.logic.is_duration_pred[target_predicate]
        for idx,t  in enumerate(dataset[sample_ID][target_predicate]['time']):
            if (not is_duration_pred) and dataset[sample_ID][target_predicate]['state'][idx]==0:
                # filter out 'fake' states for instant pred.
                continue
            cur_intensity = self.intensity(t, dataset, sample_ID, target_predicate)
            intensity_list.append(cur_intensity)
        if intensity_list:
            intensity_sum = torch.sum(torch.log(torch.cat(intensity_list)))
        else:
            intensity_sum = torch.tensor([0], dtype=torch.float64)
        return intensity_sum
        

    def intensity_integral(self, dataset, sample_ID, target_predicate, is_use_closed_integral=True):
        start_time = 0
        
        if self.args.dataset_name == "synthetic":
            end_time = self.args.synthetic_time_horizon
        else:
            end_time = dataset[sample_ID][target_predicate]['time'][-1]
            for i in range(self.num_predicate):
                end_time = max(end_time, dataset[sample_ID][i]['time'][-1])
        if end_time == 0:
            intensity_integral = torch.tensor([0], dtype=torch.float64)
        else:
            if is_use_closed_integral and self.args.non_negative_map == "max":
                intensity_integral = self._closed_integral(start_time, end_time, dataset, sample_ID, target_predicate)
            else:
                intensity_integral = self._numerical_integral(start_time, end_time, dataset, sample_ID, target_predicate)
        return intensity_integral
    
    def _closed_integral(self, start_time, end_time, dataset, sample_ID, target_predicate):
        """NOTE: this implementation has following assumptions:
        1) non_negative_map is max
        2) intensity alsways > 0
        i.e. lambda = wf + b
        """
        formula_ind_list = list(self.template[target_predicate].keys()) # extract formulas related to target_predicate
        
        if (sample_ID, target_predicate) in self.feature_integral_cache:
            feature_integral_list = self.feature_integral_cache[(sample_ID, target_predicate)]
        else:
            feature_integral_list = list()
            for formula_ind in formula_ind_list:
                feature_integral = self._closed_feature_integral(start_time, end_time, dataset, sample_ID, target_predicate, formula_ind)
                feature_integral_list.append(feature_integral)
            feature_integral_list = torch.tensor(feature_integral_list)
            self.feature_integral_cache[(sample_ID, target_predicate)] = feature_integral_list
        weight = self._parameters["weight"][formula_ind_list]
        base = self._parameters["base"][target_predicate]
        integral = torch.sum(torch.mul(feature_integral_list, weight)) +  base * (end_time - start_time) 
        return integral
    
    def _numerical_feature_integral(self, start_time, end_time, dataset, sample_ID, target_predicate, formula_ind):
        # collect evidence
        feature_integral_term = list()
        for t in np.arange(start_time, end_time, self.args.integral_grid):
            cur_state = self._check_state(dataset[sample_ID][target_predicate], t)  # cur_state is either 0 or 1
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
            feature_integral_term.append(feature)
        if len(feature_integral_term) > 0:
            feature_integral = torch.sum(torch.cat(feature_integral_term)) * self.args.integral_grid # approximate integral
        else:
            feature_integral = torch.tensor([0], dtype=torch.float64)
        
        return feature_integral.item()

    def _closed_feature_integral(self, start_time, end_time, dataset, sample_ID, target_predicate, formula_ind):
        template = self.template[target_predicate][formula_ind]
        BEFORE = self.logic.logic.BEFORE
        EQUAL = self.logic.logic.EQUAL
        DT = self.args.time_tolerence
        neighbor_ind, neighbor_combination, target_ind_in_predicate, time_template, formula_effect = template['neighbor_ind'],template['neighbor_combination'], template['target_ind_in_predicate'], template['time_template'], template['formula_effect']
        data = dataset[sample_ID][target_predicate]
        ta_tn_count = list()
        # ta_tn_count is a list of (t_a, t_n, count)
        # where t_a = float, time of pred A
        # t_n = float, time of latest pred
        # count = int, number of such combination (not used in current version)
        if len(neighbor_ind) == 1:
            #t_a == t_n, where t_n is the latest body pred.
            t_a_idx = neighbor_ind[0]
            mask = dataset[sample_ID][t_a_idx]['state'] == neighbor_combination[0]
            t_a_array = dataset[sample_ID][t_a_idx]['time'][mask]
            time_rel = time_template[1]
            for t_a in t_a_array:
                if time_rel == BEFORE:
                    ta_tn_count.append((t_a, min(t_a+DT,end_time), end_time, 1)) #integrate from t_a+DT to end_time. Use min(., end_time) to avoid overbounding.
                elif time_rel == EQUAL:
                    ta_tn_count.append((t_a, t_a, min(t_a+DT,end_time), 1))  #integrate from t_a to t_a+DT. Use min(., end_time) to avoid overbounding.

        elif len(neighbor_ind) == 2:
            #t_a != t_n, and len(body) == 2 
            t_a_idx, t_n_idx = neighbor_ind
            mask_a = dataset[sample_ID][t_a_idx]['state'] == neighbor_combination[0]
            t_a_array = dataset[sample_ID][t_a_idx]['time'][mask_a]
            mask_n = dataset[sample_ID][t_n_idx]['state'] == neighbor_combination[1]
            t_n_array = dataset[sample_ID][t_n_idx]['time'][mask_n]
            time_rel = time_template[1]
            
            for t_a, t_n in itertools.product(t_a_array, t_n_array):
                if time_rel == BEFORE and t_n - t_a >= self.args.time_tolerence:
                    t_s = max(t_a,t_n)
                    ta_tn_count.append( (t_a, min(t_s+DT, end_time), end_time, 1))
                elif time_rel == EQUAL and not abs(t_n - t_a) <= self.args.time_tolerence:
                    t_s = max(t_a,t_n)
                    ta_tn_count.append( (t_a, t_s, min(t_s+DT, end_time), 1))
        else:
            # Not implemented for long rules, use numerical instead. 
            return self._numerical_feature_integral(start_time, end_time, dataset, sample_ID, target_predicate, formula_ind)
        is_duration_pred = self.logic.logic.is_duration_pred[target_predicate]
        data = dataset[sample_ID][target_predicate]
        integral = self._closed_formula_effect_integral(ta_tn_count, end_time, data, is_duration_pred, formula_effect)
        return integral
        
    def _closed_formula_effect_integral(self, ta_tn_count, end_time, data, is_duration_pred, formula_effect):
        fe_integral = 0
        D = self.args.time_decay_rate
        if not is_duration_pred:
            state = data['state'][-1]
            fe_sign = formula_effect[state]
            for ta,ts,te,count in ta_tn_count:
                # integral from tn to end_time
                fe_integral_term = (np.exp(D*(ta - ts)) - np.exp(D*(ta - te)))/D
                fe_integral += fe_integral_term * count * fe_sign 
        else:
            
            for ta,ts,te,count in ta_tn_count:
                # integral from tn to end_time
                mask = data['time'] > ts  # NOTE: assume tn is earier than target
                time_list = data['time'][mask].tolist()
                state_list = data['state'][mask].tolist()
                # add virtual event: ts, for easy calculation
                time_list.insert(0, ts)
                state = self._check_state(data,ts)
                state_list.insert(0, state)
                # add virtual event: te, for easy calculation
                time_list.append(te)
                state = self._check_state(data, te)
                state_list.append(state)
                for i in range(len(time_list)-1):
                    ts_ = time_list[i]
                    te_ = time_list[i+1]
                    state = state_list[i]
                    fe_sign = formula_effect[state]
                    fe_integral_term = (np.exp(D*(ta - ts_)) - np.exp(D*(ta - te_)))/D
                    fe_integral += fe_integral_term * count * fe_sign 
        return fe_integral
                

    def _numerical_integral(self, start_time, end_time, dataset, sample_ID, target_predicate):
        intensity_integral_term = list()
        for t in np.arange(start_time, end_time, self.args.integral_grid):
            cur_intensity = self.intensity(t, dataset, sample_ID, target_predicate)
            intensity_integral_term.append(cur_intensity)
        if len(intensity_integral_term) > 0:
            intensity_integral = torch.sum(torch.cat(intensity_integral_term)) * self.args.integral_grid # approximate integral
        else:
            intensity_integral = torch.tensor([0], dtype=torch.float64)
        return intensity_integral


    def log_likelihood(self, dataset, sample_ID_batch):
        log_likelihood = torch.tensor([0], dtype=torch.float64)
        for sample_ID in sample_ID_batch:
            for target_predicate in self.args.target_predicate:
                intensity_log_sum = self.intensity_log_sum(dataset, sample_ID, target_predicate)
                intensity_integral = self.intensity_integral(dataset, sample_ID, target_predicate)
                log_likelihood += intensity_log_sum - intensity_integral
        return log_likelihood
        
    def predict_survival(self):
        pass