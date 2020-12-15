"""
implement Sub-problem
"""
import sys
sys.path.append("./")
sys.path.append("../")
from collections import OrderedDict
import itertools
from typing import List,Tuple,Dict,Any

import numpy as np
import cvxpy as cp
import torch

from logic import Logic
from model.point_process import Point_Process

class Sub_Problem:
    """Sub problem of Column Generation.
    """
    def __init__(self, args):
        self.args = args
        self.pp = Point_Process(self.args)
        self.pp_only_new_rule = Point_Process(self.args)
        self.logic = self.get_empty_logic()
    
    def set_logic_and_param(self, logic, w, b, lambda_):
        self.pp.set_logic(logic)
        self.pp.set_parameters(w=w, b=b, requires_grad=False)
        self.w, self.b, self.lambda_ = w, b, lambda_ 
        self.logic = logic
        self.template = {t:self.logic.get_template(t) for t in self.args.target_predicate}

    def get_empty_logic(self):
        logic = Logic(self.args)
        for i in range(logic.logic.num_formula):
            logic.delete_rule(rule_idx=0)
        return logic 
    
    def get_init_logic(self):
        """generate logic with a random rule. Used as init logic in CG.
        """
        target_predicate = self.args.target_predicate[0]
        new_rule_triplet = next(self.generate_new_rule(target_predicate))
        logic = self.get_empty_logic()
        logic.add_rule(*new_rule_triplet)
        return logic

    def generate_new_rule(self, target_predicate):
        # NOTE: In this func we assume target pred is the last pred in list.
        # Returns new_rule_triplet containing 3 np arrays: 
        #   1)logic_rule: signs of each pred
        #   2)time_template: time relation between each pred with first body pred
        #   3)R_arrray: idx of selected pred.
        num_predicate = self.logic.logic.num_predicate
        for rule_len in range(2, num_predicate+1):
            if rule_len == 2: #len_2 rules are created from scratch
                for neighbor_idx in range(num_predicate):
                    if neighbor_idx >= target_predicate: #target pred is the last
                        break 
                    R_arrray = np.zeros(num_predicate)
                    R_arrray[[neighbor_idx,target_predicate]] = 1
                    for neighbor_sign in [0,1]:
                        for target_sign in [0,1]:
                            logic_rule = np.array([neighbor_sign, target_sign])
                            for time_relation in [self.logic.logic.BEFORE, self.logic.logic.EQUAL]:
                                time_template = np.array([0, time_relation])
                                if not self._check_repeat(logic_rule, time_template, R_arrray):
                                    yield logic_rule, time_template, R_arrray

            else: # longer rules are generated from existing rules.
                for R_array_ in self.logic.R_arrray.T:
                    for i in range(num_predicate):
                        if R_arrray_[i] == 0: #extend from this rule
                            idx = np.sum(R_arrray_[:i]) # the idx of new predicate in the rule.
                            if idx == 0: #not allow to insert at beginning.
                                continue
                            time_template_ = self.logic.get_time_template(formula_ind=idx)
                            logic_rule_ = self.logic.get_logic_rule(formula_ind=idx)
                            for sign in [0,1]:
                                logic_rule = np.insert(logic_rule_, idx, sign)
                                for time_relation in [self.logic.logic.BEFORE, self.logic.logic.EQUAL]:
                                    time_template = np.insert(time_template, idx, time_relation)
                                    if not self._check_repeat(logic_rule, time_template, R_arrray):
                                        yield logic_rule, time_template, R_arrray
    
    def _check_repeat(self, rule, time_template, R_arrray):
        for formula_ind in range(self.logic.logic.num_formula):
            if (self.logic.logic.R_matrix[:,formula_ind] == R_arrray).all():
                time_template = self.logic.get_time_template(formula_ind)
                logic_rule = self.logic.get_logic_rule(formula_ind)
                if (time_template == time_template).all() and (logic_rule == rule).all():
                    return True
        return False
    
    
    def get_feature_list(self, new_rule_triplet, dataset, sample_ID, target_predicate):
        '''returns list of feature of new rule'''
        logic =  self.get_empty_logic()
        logic.add_rule(*new_rule_triplet)
        self.pp_only_new_rule.set_logic(logic)

        feature_list = list()
        is_duration_pred =  self.logic.logic.is_duration_pred[target_predicate]
        for idx,t  in enumerate(dataset[sample_ID][target_predicate]['time']):
            if (not is_duration_pred) and dataset[sample_ID][target_predicate]['state'][idx]==0:
                # filter out 'fake' states for instant pred.
                continue
            feature = self.pp_only_new_rule.get_feature(t, dataset, sample_ID, target_predicate)
            feature_list.append(feature[0])
        return torch.tensor(feature_list)

    
    def get_feature_integral(self, new_rule_triplet, dataset, sample_ID, target_predicate):
        '''returns feature integral of new rule'''
        formula_ind = 0
        start_time = 0
        logic =  self.get_empty_logic()
        logic.add_rule(*new_rule_triplet)
        self.pp_only_new_rule.set_logic(logic)
        end_time = self.pp_only_new_rule.get_end_time(dataset, sample_ID, target_predicate)
        feature_integral = self.pp_only_new_rule._closed_feature_integral(start_time, end_time, dataset, sample_ID, target_predicate, formula_ind)
        return feature_integral
        

    def get_feature_sum_list(self, dataset, sample_ID, target_predicate):
        feature_sum_list = list()
        is_duration_pred =  self.logic.logic.is_duration_pred[target_predicate]
        for idx,t  in enumerate(dataset[sample_ID][target_predicate]['time']):
            if (not is_duration_pred) and dataset[sample_ID][target_predicate]['state'][idx]==0:
                # filter out 'fake' states for instant pred.
                continue
            formula_ind_list = list(self.template[target_predicate].keys()) # extract formulas related to target_predicate
            #print("self.pp._parameters", self.pp._parameters)
            weight = self.pp._parameters["weight"][formula_ind_list]
            feature_list = self.pp.get_feature(t, dataset, sample_ID, target_predicate)
            feature_sum = torch.sum(torch.mul(feature_list, weight))
            feature_sum_list.append(feature_sum)
        return torch.tensor(feature_sum_list)
    
    def objective(self, rule_complexity, sample_ID_batch, feature_sum_list, feature_list, feature_integral):
        term_1 = 0
        for sample_ID in sample_ID_batch:
            #print("feature_list[sample_ID] = ", feature_list[sample_ID])
            #print("feature_sum_list[sample_ID] = ", feature_sum_list[sample_ID])
            # add 1e-100 to avoid zero-div error
            term_1 -= torch.sum(torch.div(feature_list[sample_ID] + 1e-100, feature_sum_list[sample_ID] + 1e-100))
        term_2 = - feature_integral
        term_3 = self.lambda_ * rule_complexity
        #print("term_1 =", term_1)
        #print("term_2 =", term_2)
        #print("term_3 =", term_3)
        objective = term_1 + term_2 + term_3
        return objective

    def iter(self, dataset, sample_ID_batch):
        best_obj = 0
        best_rule = None
        for target_predicate in self.args.target_predicate:
            feature_sum_list = dict()
            for sample_ID in sample_ID_batch:
                feature_sum_list[sample_ID] = self.get_feature_sum_list(dataset, sample_ID, target_predicate)
            for new_rule_triplet in self.generate_new_rule(target_predicate):
                print("new_rule_triplet =", new_rule_triplet)
                feature_list = dict()
                feature_integral = 0
                for sample_ID in sample_ID_batch:
                    feature_integral += self.get_feature_integral(new_rule_triplet, dataset, sample_ID, target_predicate)
                    feature_list[sample_ID] = self.get_feature_list(new_rule_triplet, dataset, sample_ID, target_predicate)
                rule_complexity = np.sum(new_rule_triplet[2]) # rule_complexity = sum(R_arrray) = length of rule
                obj = self.objective(rule_complexity, sample_ID_batch, feature_sum_list, feature_list, feature_integral)
                
                
                print("obj =", obj.item())
                
                if obj < best_obj:
                    best_obj = obj
                    best_rule = new_rule_triplet
        return best_rule


if __name__ == '__main__':
    # simple test
    from utils.args import get_args
    args = get_args()
    args.target_predicate = [1]
    logic = Logic(args)
    init_rand_rule = (np.array([1, 1]), np.array([   0, 1000]), np.array([1., 1.]))
    logic.delete_rule(0)
    logic.add_rule(*init_rand_rule)

    w = torch.ones(logic.logic.num_formula)
    b = torch.ones(logic.logic.num_predicate)
    lambda_ = 1.0
    target_predicate = 1
    sp = Sub_Problem(args, logic)
    for i in sp.generate_new_rule(target_predicate=target_predicate):
        ##print(i)
        pass
    pred0 = {"time":np.array([0, 0.9, 0.9, 1.9, 1.9]), "state":np.array([0,1,0,1,0])}
    pred1 = {"time":np.array([0, 1, 1, 2, 2]), "state":np.array([0,1,0,1,0])}
    dataset = {0:{0:pred0, 1:pred1}}
    sample_ID_batch = [0]

    sp.iter(dataset, sample_ID_batch, target_predicate)

    







