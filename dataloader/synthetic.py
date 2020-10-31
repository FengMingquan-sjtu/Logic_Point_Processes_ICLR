import sys
sys.path.extend(['./','../'])
from typing import List,Tuple,Dict,Any

import numpy as np
from numpy.random import default_rng
import torch


from logic import Logic
from model.point_process import Point_Process
from utils.data_vis import draw_event_intensity

class Synthetic:
    def __init__(self,args):
        self.args = args
        self.model = Point_Process(args=args)
        self.logic = self.model.logic
        self.num_predicate = self.logic.logic.num_predicate
        self.num_formula = self.logic.logic.num_formula
        #self.model._parameters["weight"] = torch.rand(self.num_formula) * 0.2
        self.model._parameters["weight"] = torch.ones(self.num_formula) * args.synthetic_weight
        #self.model._parameters["base"] = torch.rand(self.num_predicate) * 0.2
        self.model._parameters["base"] = torch.ones(self.num_predicate) * args.synthetic_base  
        print("=>Generating synthetic dataset.")
        print("=>Weight = ", self.model._parameters["weight"])
        print("=>Base = ", self.model._parameters["base"]) 
    
    def get_dataset(self,is_train:bool, seed:int=None)->Dict:
        """preprocess dataset.
        Args:
            is_train: bool,
                if is_train then return training data
                else return testing data
            seed: int,
                random seed, if None then use default random seed.
        Returns:
            dataset: nested dict,
                dataset[sample_ID][predicate_ID] = {'time':int list,'state':int list} 
        """
        
        if is_train:
            ub = self.args.synthetic_training_sample_num
            dataset = self.generate_data(sample_ID_lb = 0, sample_ID_ub = ub, seed=seed)
        else:
            lb = self.args.synthetic_training_sample_num
            ub = lb + self.args.synthetic_testing_sample_num
            dataset = self.generate_data(sample_ID_lb = lb, sample_ID_ub = ub, seed=seed)
    
        return dataset
    
    def _get_t_m(self, t, dataset, sample_ID, target_predicate) -> float:
        if self.args.synthetic_logic_name == "hawkes":
            is_positive_effect = True
        elif self.args.synthetic_logic_name == "self-correcting":
            is_positive_effect = False
        else:
            feature_list = self.model.get_feature(t, dataset, sample_ID, target_predicate)
            formula_ind_list = list(self.model.template[target_predicate].keys()) # extract formulas related to target_predicate
            weight = self.model._parameters["weight"][formula_ind_list]
            is_positive_effect = (torch.sum(torch.mul(feature_list, weight)) >= 0)
        # since all rules are decaying in same rate, the sign of f*w won't change.
        if is_positive_effect: # if f*w>0, then intensity keeps decreasing with time, thus max intensity point is current.
            t_m = t
        else: # if f*w<0, then intensity keeps increasing with time, then max intensity point is the furthest point.
            t_m = t + self.args.synthetic_time_horizon
        return t_m
    
    def add_new_event(self,data:Dict,t:float, pred_idx:int)->float:
        is_duration_pred=self.logic.logic.is_duration_pred[pred_idx]
        data["time"].append(t)
        last_state = data["state"][-1]
        data["state"].append(1-last_state)
        if not is_duration_pred:
            data["time"].append(t)
            data["state"].append(last_state)
        t += (self.args.time_tolerence * 1.1 ) #add time_tolerence to allow BEFORE captures this event.
        return t

    def generate_data(self, sample_ID_lb, sample_ID_ub, seed):
        """
        generate event data using Ogata’s modified thinning algorithm
        Args:
            sample_ID_lb, sample_ID_ub: int, lower bound and upper bound of sample_ID.
                ID \in [lb, ub) .
        Returns:
            dataset
        """
        dataset = dict()
        
        predicate_mapping = self.logic.logic.predicate_mapping

        for sample_ID in range(sample_ID_lb, sample_ID_ub):
            #### begin initialization  ####
            dataset[sample_ID] = dict() 
            for predicate_ID in range(self.num_predicate):
                dataset[sample_ID][predicate_ID] = {'time':[0,],'state':[0,]}
                if predicate_ID in predicate_mapping.keys():
                    for p in predicate_mapping[predicate_ID]:
                        # mapped (copied) predicate 
                        # mapped preds should always keep same with the real preds.
                        # thus only copy the reference of dict, but not copy the data of dict. 
                        dataset[sample_ID][p] = dataset[sample_ID][predicate_ID]
            #### end initialization  ####

            #### begin simulation  ####
            # Ogata's thinning    
            rng = default_rng(seed)
            cur_time = 0
            indep_pred_idx = None # independent pred index.
            while cur_time < self.args.synthetic_time_horizon:
                if self.logic.logic.independent_predicate:
                    indep_pred = self.logic.logic.independent_predicate
                    intensity_indep = self.args.intensity_indep_pred * len(indep_pred)
                    dwell_time_indep = rng.exponential(scale=1.0/ intensity_indep)
                    t_l = cur_time + dwell_time_indep
                    if t_l > self.args.synthetic_time_horizon:
                        break
                    indep_pred_idx = rng.choice(indep_pred, 1).item()
                    # not using return value (time) of add_new_event
                else:
                    t_l = self.args.synthetic_time_horizon
                    
                
                while cur_time < t_l:
                    intensity_m = np.zeros(shape = self.num_predicate)
                    # to guarantee that intensity_m >= intensity_
                    # NOTE: calculation of t_m is difficult for multiple target preds
                    # thus we use target_predicate[0] to get an approximated t_m.
                    t_m = self._get_t_m(cur_time, dataset, sample_ID, self.args.target_predicate[0])
                    for p in self.args.target_predicate:
                        intensity_m[p]= self.model.intensity(t = t_m, dataset = dataset, sample_ID = sample_ID, target_predicate = p) 
                    # only target preds update intensity, other preds won't be sampled

                    # the next event
                    dwell_time = rng.exponential(scale=1.0/ np.sum(intensity_m))
                    
                    pred_idx = rng.choice(self.num_predicate, 1, p=intensity_m / np.sum(intensity_m)).item()
                    # since copied predicates have 0 probabililty, they won't be sampeld.
                    # only the real predicates will be sampled.

                    cur_time += dwell_time# time moves forward, no matter this dwell_time is accepted or not.
                    if cur_time > t_l:
                        cur_time = t_l #key for indep-pred logic.
                        break
                    
                    #drop (thinning)
                    intensity_  = self.model.intensity(t =  cur_time, dataset = dataset, sample_ID = sample_ID, target_predicate = pred_idx)
                    accept_ratio = intensity_ / intensity_m[pred_idx]
                    #if accept_ratio > 1:
                    #    print(accept_ratio)
                    if rng.random() < accept_ratio: 
                        cur_time = self.add_new_event(data=dataset[sample_ID][pred_idx], t=cur_time, pred_idx=pred_idx)    
                if not indep_pred_idx is None: # add independent event, after while loop.
                    #print("add indep")
                    #print("cur_time =",cur_time)
                    #print("t_l =", t_l)
                    self.add_new_event(data=dataset[sample_ID][indep_pred_idx], t=t_l, pred_idx=indep_pred_idx)

            for predicate_ID in range(self.num_predicate):
                dataset[sample_ID][predicate_ID]['time'] = np.array(dataset[sample_ID][predicate_ID]['time'])
                dataset[sample_ID][predicate_ID]['state'] = np.array(dataset[sample_ID][predicate_ID]['state'])           
        return dataset
    
    def draw_dataset(self):
        dataset = self.get_dataset(is_train=1)
        def f(t):
            return self.model.intensity(t = t, dataset = dataset, sample_ID = 0, target_predicate = self.args.target_predicate[0]) 
        name = "synthetic_" + self.args.synthetic_logic_name 
        f_list = [f]
        if self.args.synthetic_logic_name in ["hawkes","self_correcting"]:
            data = dataset[0][1]
            mask = np.array(data['state'])==1
            time_array = np.array(data["time"])[mask]
            time_array_list = [time_array]
        elif self.args.synthetic_logic_name in ["a_then_b","a_then_c_or_b_then_c", "a_and_b_then_c"]:
            time_array_list = list()
            for i in range(self.num_predicate):
                time_array_list.append(np.array(dataset[0][i]["time"]))
        else:
            raise ValueError
        draw_event_intensity(time_array_list, name, f_list)




if __name__ == "__main__":
    from utils.args import get_args
    args = get_args()
    args.dataset_name = "synthetic"
    args.synthetic_logic_name = "a_and_b_then_c" #"a_then_c_or_b_then_c"#"a_then_b"#"hawkes"#"self_correcting"

    if args.synthetic_logic_name in ["a_then_c_or_b_then_c", "a_and_b_then_c"]:
        args.target_predicate = [2]
    else:
        args.target_predicate = [1]
    
    args.synthetic_training_sample_num = 1
    args.synthetic_testing_sample_num = 0
    args.synthetic_time_horizon = 50
    args.synthetic_weight = 0.2
    args.synthetic_base = 0.4
    s = Synthetic(args)
    s.draw_dataset()
    train_data_set = s.generate_data(sample_ID_lb=0, sample_ID_ub=1, seed=1)
    #print(train_data_set)
