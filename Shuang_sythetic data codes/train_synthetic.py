import datetime
import os

import numpy as np
import torch

from logic_learning import Logic_Learning_Model
from utils import redirect_log_file, Timer

def get_data(dataset_id, num_sample):
    dataset_path = './data/data-{}.npy'.format(dataset_id)
    print("dataset_path is ",dataset_path)
    dataset = np.load(dataset_path, allow_pickle='TRUE').item()
    if len(dataset.keys())> num_sample: 
        dataset = {i:dataset[i] for i in range(num_sample)}
    num_sample = len(dataset)
    training_dataset = {i: dataset[i] for i in range(int(num_sample*0.8))}
    testing_dataset = {i: dataset[int(num_sample*0.8)+i] for i in range(int(num_sample*0.2))}

    
    print("sample num is ", num_sample)
    print("training_dataset size=", len(training_dataset))
    print("testing_dataset size=", len(testing_dataset))
    return training_dataset, testing_dataset

def get_model(dataset_id):
    from generate_synthetic_data import get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5,get_logic_model_6,get_logic_model_7,get_logic_model_8,get_logic_model_9,get_logic_model_10,get_logic_model_11,get_logic_model_12,get_logic_model_13,get_logic_model_14,get_logic_model_15,get_logic_model_16,get_logic_model_17
    logic_model_funcs = [None,get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5,get_logic_model_6,get_logic_model_7,get_logic_model_8,get_logic_model_9,get_logic_model_10,get_logic_model_11,get_logic_model_12,get_logic_model_13,get_logic_model_14,get_logic_model_15,get_logic_model_16,get_logic_model_17]
    m, _ = logic_model_funcs[dataset_id]()
    model = m.get_model_for_learn()
    return model

def fit(dataset_id, num_sample, worker_num=8, num_iter=5, use_cp=False, rule_set_str = None, algorithm="BFS"):
    t  = datetime.datetime.now()
    print("Start time is", t ,flush=1)
    if not os.path.exists("./model"):
        os.makedirs("./model")    
    #get model
    model = get_model(dataset_id)
    #set initial rules if required
    if rule_set_str:
        set_rule(model, rule_set_str)
    #get data
    training_dataset, testing_dataset =  get_data(dataset_id, num_sample)

    #set model hyper params
    model.batch_size_grad = num_sample #use all sample for grad
    model.batch_size_cp = num_sample
    model.num_iter = num_iter
    model.use_cp = use_cp
    model.worker_num = worker_num
    
    if dataset_id in [5,12]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.05
        model.strict_weight_threshold= 0.1
    elif dataset_id in [6,13]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.2
        model.strict_weight_threshold= 0.5
    elif dataset_id in [7,14,17]:
        model.max_rule_body_length = 3
        model.max_num_rule = 20
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3
    elif dataset_id in [8,15]:
        model.max_rule_body_length = 2
        model.max_num_rule = 20
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3
    elif dataset_id in [4,9,10,11,16]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3


    if algorithm == "DFS":
        with Timer("DFS") as t:
            model.DFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag = dataset_id)
    elif algorithm == "BFS":
        with Timer("BFS") as t:
            model.BFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag = dataset_id)
    
    print("Finish time is", datetime.datetime.now())
 

def run_expriment_group(dataset_id):
    #DFS
    fit(dataset_id=dataset_id, num_sample=600, worker_num=12, num_iter=12, algorithm="DFS")
    fit(dataset_id=dataset_id, num_sample=1200, worker_num=12, num_iter=12, algorithm="DFS")
    fit(dataset_id=dataset_id, num_sample=2400, worker_num=12, num_iter=12, algorithm="DFS")

    #BFS 
    fit(dataset_id=dataset_id, num_sample=600, worker_num=12, num_iter=12, algorithm="BFS")
    fit(dataset_id=dataset_id, num_sample=1200, worker_num=12, num_iter=12, algorithm="BFS")
    fit(dataset_id=dataset_id, num_sample=2400, worker_num=12, num_iter=12, algorithm="BFS")

if __name__ == "__main__":
    redirect_log_file()

    torch.multiprocessing.set_sharing_strategy('file_system') #fix bug#78
    
    #run_expriment_group(dataset_id=13)
    #run_expriment_group(dataset_id=14)
    #run_expriment_group(dataset_id=15)
    #run_expriment_group(dataset_id=16)
    #run_expriment_group(dataset_id=17)

    fit(dataset_id=15, num_sample=2400, worker_num=12, num_iter=12, algorithm="BFS")