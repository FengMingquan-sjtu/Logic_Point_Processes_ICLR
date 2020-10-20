import sys
sys.path.extend(['./','../'])

import numpy as np
from numpy.random import default_rng
import torch


from logic import Logic
from model.temporal_reason import TemporalReason
from debug.hawkes import Hawkes # used to debug

class Synthetic:
    def __init__(self,args):
        self.args = args
        
        self.model = TemporalReason(
            args = self.args,
            train_dataset = None, 
            test_dataset = None, 
            template = dict(),)
            
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

        self.__update_template()  
        #print(self.model.template) 
        self.debug_model = Hawkes(args)


    def __update_template(self):
        """calculate templates of model.
        """
        # all predicates, including non-target, requires template
        for p in range(self.num_predicate):
            self.model.template[p] = self.logic.get_template(target_predicate_ind=p)
    
    def get_dataset(self,is_train, seed=None):
        """preprocess dataset.
        Args:
            is_train: bool,
                if is_train then return training data
                else return testing data
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
    
    def _get_t_m(self, t, dataset, sample_ID, target_predicate):
        feature_list = self.model.get_feature(t, dataset, sample_ID, target_predicate)
        formula_ind_list = list(self.template[target_predicate].keys()) # extract formulas related to target_predicate
        weight = self._parameters["weight"][formula_ind_list]
        is_positive_effect = (torch.sum(torch.mul(feature_list, weight)) >= 0)
        # since all rules are decaying in same rate, the sign of f*w won't change.
        if is_positive_effect: # if f*w>0, then intensity keeps decreasing with time, thus max intensity point is current.
            t_m = t
        else: # if f*w<0, then intensity keeps increasing with time, then max intensity point is the furthest point.
            t_m = t + self.args.synthetic_time_horizon
        return t_m
    
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
            while cur_time < self.args.synthetic_time_horizon:
                if self.logic.independent_predicate:
                    indep_pred = self.logic.independent_predicate
                    intensity_indep = self.args.intensity_indep_pred * len(indep_pred)
                    dwell_time_indep = rng.exponential(scale=1.0/ intensity_indep)
                    t_l = cur_time + dwell_time_indep
                    if t_l > self.args.synthetic_time_horizon:
                        break
                    pred_idx = rng.choice(indep_pred, 1)
                    pred_idx = pred_idx.item()
                    # TODO append this pred to dataset...
                    # maybe define add_new_event() function.
                    # ...
                    
                    
                else:
                    t_l = self.args.synthetic_time_horizon
                intensity_m = np.zeros(shape = self.num_predicate)

                # to guarantee that intensity_m >= intensity_
                t_m = self._get_t_m(cur_time)
                for p in self.args.target_predicate:
                    intensity_m[p]= self.model.intensity(t = t_m, dataset = dataset, sample_ID = sample_ID, target_predicate = p) 
                # only real preidicates update intensity.
                # the copied predicates always have 0 intensity (thus they won't be sampled).

                # the next event
                dwell_time = rng.exponential(scale=1.0/ np.sum(intensity_m))
                
                pred_idx = rng.choice(self.num_predicate, 1, p=intensity_m / np.sum(intensity_m))
                pred_idx = pred_idx.item()

                # since copied predicates have 0 probabililty, they won't be sampeld.
                # only the real predicates will be sampled.

                cur_time += dwell_time# time moves forward, no matter this dwell_time is accepted or not.
                if cur_time > min(t_l, self.args.synthetic_time_horizon):
                    break
                
                #drop (thinning)
                intensity_  = self.model.intensity(t =  cur_time, dataset = dataset, sample_ID = sample_ID, target_predicate = pred_idx)
                #debug_int = self.debug_intensity(t =  cur_time, dataset = dataset, sample_ID = sample_ID)
                #if abs(intensity_-debug_int) > 0.01:   
                #    print("ERROR 2")

                accept_ratio = intensity_ / intensity_m[pred_idx]
                #if accept_ratio > 1:
                #    print(accept_ratio)
                

                if rng.random() < accept_ratio: 

                    dataset[sample_ID][pred_idx]["time"].append(cur_time)
                    dataset[sample_ID][pred_idx]["state"].append(1)

                    # must add time_tolerence
                    # otherwise, next intensity_m can not count current event, which leads to accept_ratio > 1.
                    #cur_time += (self.args.time_tolerence *1.01 )
                    cur_time += self.args.time_tolerence * 1.0 # must be exactly time_tolerence.

                    dataset[sample_ID][pred_idx]["time"].append(cur_time)
                    dataset[sample_ID][pred_idx]["state"].append(0)
                    cur_time += (self.args.time_tolerence * 0.1 ) #must add this line
                else:
                    continue

                
        return dataset
    
    def draw_dataset(self):
        dataset = self.get_dataset(is_train=1)
        #print(dataset)
        data = dataset[0][1]
        mask = np.array(data['state'])==1
        time_array = np.array(data["time"])[mask]
        def f(t):
            return self.model.intensity(t = t, dataset = dataset, sample_ID = 0, target_predicate = 1) 
        name = "synthetic_" + self.args.synthetic_logic_name 

        h = Hawkes(self.args)
        h.draw(time_array, name, f)




if __name__ == "__main__":
    from utils.args import get_args
    
    args = get_args()
    args.dataset_name = "synthetic"
    args.target_predicate = [1]
    args.synthetic_logic_name = "self_correcting"
    args.synthetic_training_sample_num = 10
    args.synthetic_testing_sample_num = 10
    args.synthetic_time_horizon = 50
    s = Synthetic(args)
    s.draw_dataset()
    train_data_set = s.get_dataset(1)
    print(train_data_set)
    
    #print(test_data_set)
    #event_cnt = 0
    #for ID,info in test_data_set.items():
    #    event_cnt += len(info[1]['time']) / 2  # divided by 2, since each event has 2 states (times).
    #avg_event_cnt = event_cnt / len(test_data_set.keys())
    #print(avg_event_cnt)