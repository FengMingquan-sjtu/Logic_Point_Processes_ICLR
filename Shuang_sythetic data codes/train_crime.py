import datetime
import os

import numpy as np
import torch

from logic_learning import Logic_Learning_Model
from utils import redirect_log_file, Timer, get_data, get_args

def get_model(model_name, dataset_name):
    if model_name == "crime":
        model = Logic_Learning_Model(head_predicate_idx=[8])
        model.predicate_set= [0, 1, 2, 3, 4, 5, 6, 7, 8] # the set of all meaningful predicates
        model.predicate_notation = ['SUMMER', 'WINTER', 'WEEKEND', 'EVENNING', 'NIGHT',  'A','B', 'C', 'D']
        model.body_pred_set =  model.predicate_set
        model.static_pred_set = [0, 1, 2, 3, 4]
        model.instant_pred = [5, 6, 7, 8]
        
        model.max_rule_body_length = 3
        model.max_num_rule = 20
        model.weight_threshold = 0.001
        model.strict_weight_threshold= 0.005
        model.gain_threshold = 0.001
        model.low_grad_threshold = 0.001
        model.batch_size_grad = 100

    if dataset_name.endswith("day"):
        model.time_window = 10
        model.Time_tolerance = 1
        model.decay_rate = 1
        model.batch_size = 64
        model.integral_resolution = 0.1

    elif dataset_name.endswith("week"):
        model.time_window = 7 * 24
        model.Time_tolerance = 12
        model.decay_rate = 0.01
        model.batch_size = 8
        model.integral_resolution = 1
    
    elif dataset_name.endswith("month"):
        model.time_window = 7 * 24
        model.Time_tolerance = 12
        model.decay_rate = 0.01
        model.batch_size = 4
        model.integral_resolution = 1
    
    elif dataset_name.endswith("year"):
        model.time_window = 7 * 24
        model.Time_tolerance = 12
        model.decay_rate = 0.01
        model.batch_size = 1
        model.integral_resolution = 1
    
    return model

def fit(model_name, dataset_name, num_sample, worker_num=8, num_iter=5, use_cp=False, rule_set_str = None, algorithm="BFS"):
    print("Start time is", datetime.datetime.now(),flush=1)

    if not os.path.exists("./model"):
        os.makedirs("./model")
    
    #get model
    model = get_model(model_name, dataset_name)

    #set initial rules if required
    if rule_set_str:
        set_rule(model, rule_set_str)
        

    #get data
    dataset =  get_data(dataset_name, num_sample)

    #set model hyper params
    model.batch_size_grad = num_sample #use all sample for grad
    model.batch_size_cp = num_sample
    model.num_iter = num_iter
    model.use_cp = use_cp
    model.worker_num = worker_num
    
    
    
    


    if algorithm == "DFS":
        with Timer("DFS") as t:
            model.DFS(model.head_predicate_set[0], dataset, tag="DFS_"+dataset_name)
    elif algorithm == "BFS":
        with Timer("BFS") as t:
            model.BFS(model.head_predicate_set[0], dataset, tag="BFS_"+dataset_name)
    
    print("Finish time is", datetime.datetime.now())
 

def run_expriment_group(dataset_name):
    #downtown districts
    #DFS
    fit(model_name="crime", dataset_name=dataset_name, num_sample=200, worker_num=12, num_iter=12, algorithm="DFS")
    #BFS
    fit(model_name="crime", dataset_name=dataset_name, num_sample=200, worker_num=12, num_iter=12, algorithm="BFS")



if __name__ == "__main__":
    redirect_log_file()

    torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78
    #args = get_args()
    run_expriment_group(dataset_name="crime_all_month")
    #run_expriment_group(dataset_name="crime_other_year")
    
