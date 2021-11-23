import datetime
import os
import argparse

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
    from generate_synthetic_data import get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5,get_logic_model_6,get_logic_model_7,get_logic_model_8,get_logic_model_9,get_logic_model_10,get_logic_model_11,get_logic_model_12
    logic_model_funcs = [None,get_logic_model_1,get_logic_model_2,get_logic_model_3,get_logic_model_4,get_logic_model_5,get_logic_model_6,get_logic_model_7,get_logic_model_8,get_logic_model_9,get_logic_model_10,get_logic_model_11,get_logic_model_12]
    m, _ = logic_model_funcs[dataset_id]()
    model = m.get_model_for_learn()
    return model

def fit(dataset_id, num_sample, time_limit, worker_num=8, num_epoch=5, algorithm="RAFS"):
    """Train synthetic data set, define hyper-parameters here."""
    t  = datetime.datetime.now()
    print("Start time is", t ,flush=1)
    if not os.path.exists("./model"):
        os.makedirs("./model")    
    model = get_model(dataset_id)
    training_dataset, testing_dataset =  get_data(dataset_id, num_sample)

    #set model hyper params
    model.time_limit = time_limit
    model.num_epoch = num_epoch
    model.worker_num = worker_num
    model.print_time = False
    model.weight_lr = 0.0005
    model.l1_coef = 0.1

    if model.use_exp_kernel:
        model.init_base = 0.01
        model.init_weight = 0.1
    else:
        model.init_base = 0.2
        model.init_weight = 0.1
    
    if algorithm == "Brute":
        #smaller init weight and  smaller lr
        model.init_weight = 0.01
        model.weight_lr = 0.0001

    
    if dataset_id in [2,8]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.05
        model.strict_weight_threshold= 0.1
    elif dataset_id in [3,9]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.2
        model.strict_weight_threshold= 0.5
    elif dataset_id in [4,10]:
        model.max_rule_body_length = 3
        model.max_num_rule = 20
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3
    elif dataset_id in [5,11]:
        model.max_rule_body_length = 2
        model.max_num_rule = 20
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3
    elif dataset_id in [1,6,7,12]:
        model.max_rule_body_length = 2
        model.max_num_rule = 15
        model.weight_threshold = 0.1
        model.strict_weight_threshold= 0.3
    else:
        print("Warning: Hyperparameters not set!")

    if dataset_id in [1, 6, 7, 8, 11, 12]:
        model.weight_lr = 0.0001

    if algorithm == "REFS":
        with Timer("REFS") as t:
            model.REFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag = dataset_id)
    elif algorithm == "RAFS":
        with Timer("RAFS") as t:
            model.RAFS(model.head_predicate_set[0], training_dataset, testing_dataset, tag = dataset_id)
    elif algorithm == "Brute":
        with Timer("Brute") as t:
            model.Brute(model.head_predicate_set[0], training_dataset)
    
    print("Finish time is", datetime.datetime.now())
 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_id', type=int, 
        help="an integer between 1 and 12, indicating one of 12 datasets",
        default=1,
        choices=list(range(1,13)))
    parser.add_argument('--algorithm', type=str, 
        help="which seaching scheme to use, possible choices are [RAFS,REFS,Brute].",
        default="RAFS",
        choices=["RAFS","REFS","Brute"])
    parser.add_argument('--time_limit', type=float, 
        help="maximum running time (seconds)",
        default=3600 * 24, # 24 hours
        )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system') #multi process communication strategy, depending on operating system.

    args = get_args()
    
    redirect_log_file() #redirect stdout and stderr to log files.

    fit(dataset_id=args.dataset_id, time_limit=args.time_limit, num_sample=2400, worker_num=12, num_epoch=12, algorithm=args.algorithm)
    

    