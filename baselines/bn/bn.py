import sys
sys.path.extend(["./","../","../../"])
import os

import torch
import numpy as np
import pandas as pd 
import bnlearn as bn

from logic import Logic
from dataloader import get_dataset
from utils.args import get_args
from dataset_utils import convert_temporal_to_static_data


def convert_dict_to_df(dataset):
    """
    input: stattic dataset in dict form

    output: (N*T) * num_pred dataframe
    """
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



if __name__ == "__main__":
    
    args = get_args()
    train_dataset, test_dataset = get_dataset(args)
    #print(train_dataset)
    #_test_convert_dataset()
    train_bn(train_dataset, args)