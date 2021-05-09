import datetime
import os

import numpy as np
import torch

from logic_learning import Logic_Learning_Model
from utils import redirect_log_file, Timer, get_data

def get_model(model_name):
    if model_name == "crime":
        model = Logic_Learning_Model(head_predicate_idx=[6])
        model.predicate_set= [0, 1, 2, 3, 4, 5, 6] # the set of all meaningful predicates
        model.predicate_notation = ["SUMMER", "WEEKEND",  "EVENNING", 'A','B', 'C', 'D']
        model.static_pred_set = [0, 1, 2]
        model.instant_pred = [3, 4, 5, 6]
        T_max = 24
    return model,T_max

def fit(model_name, num_sample, worker_num=8, num_iter=5, use_cp=False, rule_set_str = None, algorithm="BFS"):
    print("Start time is", datetime.datetime.now(),flush=1)

    if not os.path.exists("./model"):
        os.makedirs("./model")
    
    #get model
    model, T_max = get_model(model_name)

    #set initial rules if required
    if rule_set_str:
        set_rule(model, rule_set_str)
        

    #get data
    dataset =  get_data(model_name, num_sample)

    #set model hyper params
    model.batch_size_grad = num_sample #use all sample for grad
    model.batch_size_cp = num_sample
    model.num_iter = num_iter
    model.use_cp = use_cp
    model.worker_num = worker_num
    
    
    model.max_rule_body_length = 3
    model.max_num_rule = 20
    model.weight_threshold = 0.5
    model.strict_weight_threshold= 1.0
    


    if algorithm == "DFS":
        with Timer("DFS") as t:
            model.DFS(model.head_predicate_set[0], dataset, T_max=T_max, tag=model_name)
    elif algorithm == "BFS":
        with Timer("BFS") as t:
            model.BFS(model.head_predicate_set[0], dataset, T_max=T_max, tag=model_name)
    
    print("Finish time is", datetime.datetime.now())
 

def run_expriment_group(model_name):
    #DFS
    fit(model_name=model_name, num_sample=1500, worker_num=12, num_iter=12, algorithm="DFS")
    
    #BFS
    fit(model_name=model_name, num_sample=1500, worker_num=12, num_iter=12, algorithm="BFS")

if __name__ == "__main__":
    redirect_log_file()

    torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78
    run_expriment_group(model_name="crime")
    
