import sys
sys.path.extend(["./","../","../../"])
import os
import argparse

import torch
import numpy as np
import pandas as pd 
import bnlearn as bn
import pickle

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

def train_mimic_bn(bn_args):
    df = pd.read_csv(bn_args.train_csv_path, index_col=0)
    df = df.drop('row_id', axis=1)
    model = bn.structure_learning.fit(df, methodtype='hc', scoretype='bic')
    model = bn.parameter_learning.fit(model, df, methodtype="bayes", verbose=2)
    with open(bn_args.model_save_path, 'wb') as f:
        pickle.dump(model, f)
    
def test_mimic_bn(bn_args):
    df = pd.read_csv(bn_args.test_csv_path, index_col=0)
    df = df.drop('row_id', axis=1)
    with open(bn_args.model_save_path, 'rb') as f:
        model = pickle.load(f)
    target = ['flag','valuenum1','valuenum2']
    y = df[target]
    x = df.drop(target, axis=1)
    for index, x_row in x.iterrows():
        print(dict(x_row))
        y_hat = bn.inference.fit(model, variables=target, evidence=dict(x_row))

def get_bn_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="train",
                        help="tesk is one of [train, test]",)
    parser.add_argument('--train_csv_path', type=str, default="/Users/fmq/Downloads/sepsis_data_three_versions/sepsis_01/sepsis_01_test.csv",
                        help="input training csv file path",)
                        #mini-data at /Users/fmq/Downloads/sepsis_data_three_versions/sepsis_01/mini.csv
    parser.add_argument('--test_csv_path', type=str, default="/Users/fmq/Downloads/sepsis_data_three_versions/sepsis_01/sepsis_01_test.csv",
                        help="input testing csv file path",)
    parser.add_argument('--model_save_path', type=str, default="./trained_bn_model.pkl",
                        help="trained model saving path",)
    bn_args = parser.parse_args()
    return bn_args


if __name__ == "__main__":
    
    #args = get_args() #global args
    bn_args = get_bn_args() #bn(local) args
    #train_dataset, test_dataset = get_dataset(args)
    #print(train_dataset)
    #_test_convert_dataset()
    #train_bn(train_dataset, args)
    train_mimic_bn(bn_args)
    #test_mimic_bn(bn_args)