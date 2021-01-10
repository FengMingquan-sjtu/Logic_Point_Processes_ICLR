import sys
sys.path.extend(["./","../","../../"])

import os

import torch
import cvxpy as cp
import numpy as np
import pandas as pd 
import bnlearn as bn

from logic import Logic
from dataloader import get_dataset
from utils.args import get_args
from model.point_process import Point_Process

def convert_temporal_to_static_data(dataset, args):
    #input: 
    #dataset[sample_ID][predicate_ID]['time','state']
    #is_duration_pred: boolean list 
    #time_window: float
    is_duration_pred = Logic(args).logic.is_duration_pred
    bn_time_window = args.bn_time_window
    pp = Point_Process(args)
    static_dataset = dict()
    for sample_ID,data in dataset.items():
        static_dataset[sample_ID] = dict()
        for predicate_ID,data_ in data.items():
            max_time = data_['time'][-1]
            cur_time = 0
            time_list = list()
            if is_duration_pred[predicate_ID]: #for duration: sample at t = k * bn_time_window

                while cur_time <=  max_time:
                    cur_state = pp._check_state(data_,cur_time)
                    time_list.append(cur_state)
                    cur_time += bn_time_window
            else:
                #for instant: check whether jumps between  [k*bn_time_window, (k+1)*bn_time_window]
                max_idx = len(data_['time'])-1
                cur_idx = 0
                while cur_idx <= max_idx and cur_time <= max_time:
                    if data_['time'][cur_idx] > cur_time and data_['time'][cur_idx] <= cur_time + bn_time_window:
                        cur_state = 1
                    else:
                        cur_state = 0
                    while cur_idx <= max_idx and data_['time'][cur_idx] <= cur_time + bn_time_window :
                        cur_idx += 1
                    time_list.append(cur_state)
                    cur_time += bn_time_window
            static_dataset[sample_ID][predicate_ID] = time_list
    return static_dataset

def convert_dict_to_df(dataset):
    #print(dataset)
    new_data = {predicate_ID:[] for predicate_ID in dataset[0].keys()}
    for sample_ID,data in dataset.items():
        min_len = min([len(t) for t in data.values()])
        for predicate_ID,data_ in data.items():
            new_data[predicate_ID].extend(data_[:min_len])#align each predicates
    #print(new_data)
    df = pd.DataFrame(data=new_data)
    return df


def train_bn(dataset, args):
    static_dataset = convert_temporal_to_static_data(dataset, args)
    df = convert_dict_to_df(static_dataset)
    model = bn.structure_learning.fit(df)
    G = bn.plot(model)


def _test_convert_dataset():
    dataset = {0:{0:{'time':np.array([1,3,3.4,5,6,7,8]), 'state':np.array([0,1,0,1,0,1,0])}}}
    args = get_args()
    new_dataset = convert_temporal_to_static_data(dataset, args)
    print(new_dataset)






if __name__ == "__main__":
    
    args = get_args()
    train_dataset, test_dataset = get_dataset(args)
    #print(train_dataset)
    #_test_convert_dataset()
    train_bn(train_dataset, args)