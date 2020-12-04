"""
implement Sub-problem
"""
from collections import OrderedDict
import itertools
from typing import List,Tuple,Dict,Any

import numpy as np
import cvxpy as cp
import torch

from logic import Logic
from point_process import Point_Process

class Sub_Problem:
    """Sub problem of Column Generation.
    """
    def __init__(self, args, logic, w, b, lambda_):
        
        self.args = args
        self.logic = logic
        self.template = {t:self.logic.get_template(t) for t in args.target_predicate}

        self.num_formula = self.logic.logic.num_formula
        self.num_predicate = self.logic.logic.num_predicate

        self.w = w
        self.b = b
        self.lambda_ = lambda_

        self.pp = Point_Process(self.args)
        self.pp.set_parameters(w=w, b=b, requires_grad=False)

    def generate_new_R_array(self, target_predicate):
        R_arrray_basic = np.zeros(self.num_predicate)
        R_arrray_basic[target_predicate] = 1
        for rule_len in range(2, self.num_predicate+1):
            if rule_len == 2: #len_2 rules are created from scratch
                for neighbor_idx in range(self.num_predicate):
                    if neighbor_idx == target_predicate:
                        continue 
                    R_arrray = R_arrray_basic.copy()
                    R_arrray[neighbor_idx] = 1
                    yield R_arrray
            else: # longer rules are generated from existing rules.
                for R_array_ in self.logic.R_arrray.T:
                    for i in range(self.num_predicate):
                        if R_arrray_[i] == 0: #extend from this rule
                            R_arrray = R_array_.copy()
                            R_arrray[i] = 1
                            yield R_arrray
    



                    

                    


    
    def _check_repeat(self, rule, time_template, R_arrray):
        for formula_ind in range(self.num_formula):
            if (self.logic.R_matrix[:,formula_ind] == R_arrray).all():
                time_template = self.logic.get_time_template(formula_ind)
                logic_rule = self.logic.get_logic_rule(formula_ind)
                if (time_template == time_template).all() and (logic_rule == rule).all():
                    return True
        return False
    
    def add_new_rule(self, rule, time_template, R_arrray):
        self.logic.add_rule(rule, time_template, R_arrray)
        self.pp.set_logic(self.logic)
    
    def get_feature_list(self, dataset, sample_ID, target_predicate):
        feature_list = list()
        is_duration_pred =  self.logic.logic.is_duration_pred[target_predicate]
        for idx,t  in enumerate(dataset[sample_ID][target_predicate]['time']):
            if (not is_duration_pred) and dataset[sample_ID][target_predicate]['state'][idx]==0:
                # filter out 'fake' states for instant pred.
                continue
            feature = self.pp.get_feature(t, dataset, sample_ID, target_predicate)
            feature_list.append(feature[0])
        return feature_list

    
    def get_feature_integral(self, dataset, sample_ID, target_predicate):
        formula_ind = 0
        start_time = 0
        end_time = self.pp.get_end_time(dataset, sample_ID, target_predicate)
        feature_integral = self.pp._closed_feature_integral(start_time, end_time, dataset, sample_ID, target_predicate, formula_ind)
        return feature_integral
        

    def get_feature_sum_list(self, dataset, sample_ID, target_predicate):
        feature_sum_list = list()
        is_duration_pred =  self.logic.logic.is_duration_pred[target_predicate]
        for idx,t  in enumerate(dataset[sample_ID][target_predicate]['time']):
            if (not is_duration_pred) and dataset[sample_ID][target_predicate]['state'][idx]==0:
                # filter out 'fake' states for instant pred.
                continue
            formula_ind_list = list(self.template[target_predicate].keys()) # extract formulas related to target_predicate
            weight = self._parameters["weight"][formula_ind_list]
            feature_list = self.pp.get_feature(t, dataset, sample_ID, target_predicate)
            feature_sum = torch.sum(torch.mul(feature_list, weight))
            feature_sum_list.append(feature_sum)
        return feature_sum_list
    
    def objective(self, rule_complexity, feature_sum_list, feature_list, feature_integral):
        term_1 = - torch.sum(torch.div(feature_list, feature_sum_list))
        term_2 = - feature_integral
        term_3 = self.lambda_ * rule_complexity
        objective = term_1 + term_2 + term_3
        return objective


