import datetime
import os
import argparse
import pickle

import numpy as np
import torch
import pandas

from logic_learning import Logic_Learning_Model
from utils import redirect_log_file, Timer, get_data

def get_model():
    model = Logic_Learning_Model(head_predicate_idx=[0])
    model.predicate_notation = ["A"]
    model.predicate_set= [0]
    model.body_pred_set = [0]
    model.static_pred_set = []
    model.instant_pred_set = [0]
    
    model.max_rule_body_length = 1
    model.max_num_rule = 20
    model.weight_threshold = 0.001
    model.strict_weight_threshold= 0.002
    model.gain_threshold = 0.001
    model.low_grad_threshold = 0.001
    
    model.time_window = 10
    model.Time_tolerance = 1
    model.decay_rate = 0.1
    model.batch_size = 1
    model.integral_resolution = 0.5

    num_sample = 1

    model.batch_size_grad = num_sample #use all samples for grad
    model.num_iter = 1
    model.num_iter_final = 1
    model.use_cp = False
    model.worker_num = 10
    model.debug_mode = True

    return model

def get_data():
    return {0:{0:{"time":[0.9, 1.0, 1.1, 1.9, 2.0, 2.1], "state":[1, 1, 1, 1, 1, 1]}}}


if __name__ == "__main__":
    model = get_model()
    data = get_data()
    #model.optimize_log_likelihood_mp(head_predicate_idx=0, dataset=data)
    model.BFS(head_predicate_idx=0, training_dataset=data, testing_dataset=data, tag="debug")

